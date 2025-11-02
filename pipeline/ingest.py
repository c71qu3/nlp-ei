"""Standalone ingestion script for arXiv papers.

This reproduces the notebook ingestion workflow as a CLI/py module. It:
- downloads the recent listing for a given arXiv category
- extracts HTML pages, abstracts, and structured sections
- writes files under the repository `data/raw/` directory
- uses async/await for concurrent downloads (much faster!)

Usage (from repository root):
    python -m pipeline.ingest --n-papers 50 --category cs.AI
    python -m pipeline.ingest --n-papers 50 --category cs.AI --concurrency 10

"""
import argparse
import asyncio
import logging
import json
from pathlib import Path
import aiohttp
from bs4 import BeautifulSoup
from .utils import get_paper_urls, get_title, get_abstract, get_sections, check_file_exists


def ensure_dirs(base_path: Path):
    (base_path / "raw").mkdir(parents=True, exist_ok=True)
    (base_path / "raw" / "htmls").mkdir(exist_ok=True)
    (base_path / "raw" / "abstracts").mkdir(exist_ok=True)
    (base_path / "raw" / "parsed_sections").mkdir(exist_ok=True)


async def download_listing(arxiv_category: str, session: aiohttp.ClientSession):
    """Async download of the arXiv listing page."""
    arxiv_url = f"https://arxiv.org/list/{arxiv_category}/recent?skip=0&show=1000"
    async with session.get(arxiv_url) as resp:
        resp.raise_for_status()
        text = await resp.text()
    return text, arxiv_url


async def process_paper(url: str, session: aiohttp.ClientSession, base_path: Path, semaphore: asyncio.Semaphore):
    """Download and process a single paper (async)."""
    async with semaphore:  # Limit concurrent downloads
        html_id = url.split("/")[-1]
        full_content_file = base_path / "raw" / "htmls" / f"{html_id}.txt"
        abstract_content_file = base_path / "raw" / "abstracts" / f"{html_id}.txt"
        parsed_sections_file = base_path / "raw" / "parsed_sections" / f"{html_id}.json"

        # Skip if already downloaded
        if check_file_exists(str(full_content_file)) and check_file_exists(str(abstract_content_file)):
            logging.info("Both files for %s already downloaded, skipping.", html_id)
            return

        try:
            async with session.get(url) as resp:
                resp.raise_for_status()
                page_text = await resp.text()
        except Exception as e:
            logging.warning("Failed to download %s: %s", url, e)
            return

        if len(page_text) < 3000:
            logging.warning("Skipping %s: not enough content", url)
            return

        # Parse HTML (CPU-bound, but fast enough)
        paper_soup = BeautifulSoup(page_text, "html.parser")
        title = get_title(paper_soup)

        try:
            abstract = get_abstract(paper_soup)
        except AttributeError:
            logging.warning("Skipping %s as it does not have an abstract.", html_id)
            return

        try:
            article = paper_soup.find("html", recursive=False).find("body", recursive=False).find(
                "div", class_="ltx_page_main", recursive=False).find(
                "div", class_="ltx_page_content", recursive=False).find("article", recursive=False)
            sections = get_sections(article)
        except Exception:
            logging.warning("Error parsing html sections for %s", html_id)
            sections = []

        # Write files
        logging.info("Writing files for %s - %s", html_id, title)
        with open(full_content_file, "w", encoding="utf8") as f:
            f.write(page_text)
        with open(abstract_content_file, "w", encoding="utf8") as f:
            f.write(abstract)
        with open(parsed_sections_file, "w", encoding="utf8") as f:
            json.dump(sections, f, ensure_ascii=False, indent=2)


async def run_async(arxiv_category: str = "cs.AI", n_papers: int = 50, data_dir: str = "data", concurrency: int = 10):
    """Main async pipeline."""
    base_path = Path(data_dir)
    ensure_dirs(base_path)

    # Create semaphore to limit concurrent downloads
    semaphore = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession(headers={"User-Agent": "nlp-ei-bot/0.1"}) as session:
        logging.info("Fetching listing for category %s", arxiv_category)
        html_text, arxiv_url = await download_listing(arxiv_category, session)
        
        # Parse listing to get paper URLs
        main_soup = BeautifulSoup(html_text, "html.parser")
        navigation = main_soup.select("dl > dt")
        paper_urls = get_paper_urls(navigation=navigation, arxiv_url=arxiv_url, n_papers=n_papers)
        
        logging.info("Found %d paper URLs, processing with concurrency=%d", len(paper_urls), concurrency)
        
        # Process all papers concurrently
        tasks = [process_paper(url, session, base_path, semaphore) for url in paper_urls]
        await asyncio.gather(*tasks)
        
        logging.info("Pipeline complete!")


def run(arxiv_category: str = "cs.AI", n_papers: int = 50, data_dir: str = "data", concurrency: int = 10):
    """Sync wrapper for the async pipeline."""
    asyncio.run(run_async(arxiv_category, n_papers, data_dir, concurrency))


def _cli():
    parser = argparse.ArgumentParser(description="ArXiv ingestion pipeline for nlp-ei project (async)")
    parser.add_argument("--category", default="cs.AI", help="ArXiv category to scrape (e.g. cs.AI)")
    parser.add_argument("--n-papers", default=50, type=int, help="Number of papers to attempt to download")
    parser.add_argument("--data-dir", default="data", help="Base data directory to write outputs into")
    parser.add_argument("--concurrency", default=10, type=int, help="Max concurrent downloads (default: 10)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")
    run(arxiv_category=args.category, n_papers=args.n_papers, data_dir=args.data_dir, concurrency=args.concurrency)


if __name__ == "__main__":
    _cli()
