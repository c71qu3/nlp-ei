"""Utility functions extracted and hardened from the arXiv notebook.

Functions:
- get_paper_urls
- get_title
- get_abstract
- get_sections
- check_file_exists

These are intentionally small and dependency-free (only requests/bs4).
"""
from typing import List
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import os

def check_file_exists(path: str) -> bool:
    return os.path.exists(path)


def get_paper_urls(navigation, arxiv_url: str, n_papers: int) -> List[str]:
    """Return list of paper html view URLs from navigation entries.

    navigation should be a ResultSet like `main_soup.select("dl > dt")`.
    """
    paper_urls = []
    for item in navigation[:n_papers]:
        html_tag = item.select_one("a[title='View HTML'][id^='html-']")
        html_url = urljoin(arxiv_url, html_tag["href"]) if html_tag else None
        if html_url:
            paper_urls.append(html_url)
    return paper_urls


def get_title(soup: BeautifulSoup) -> str:
    """Extract <title> text from a BeautifulSoup object."""
    title_tag = soup.find("title")
    return title_tag.get_text().strip() if title_tag else ""


def get_abstract(soup: BeautifulSoup) -> str:
    """Extract abstract from arXiv HTML structure used by the notebook.

    This mirrors the notebook's approach and will raise AttributeError if
    the expected nodes are not present.
    """
    abstract_h6 = soup.find("h6", class_="ltx_title ltx_title_abstract")
    if not abstract_h6:
        # fall back: try to find a heading containing the word 'Abstract'
        candidates = soup.find_all(lambda tag: tag.name in ["h1","h2","h3","h4","h5","h6"] and 'abstract' in tag.get_text().lower())
        abstract_h6 = candidates[0] if candidates else None
    if not abstract_h6:
        raise AttributeError("Abstract heading not found")
    p = abstract_h6.find_next("p", class_="ltx_p")
    if not p:
        # fallback to next <p>
        p = abstract_h6.find_next("p")
    abstract_text = p.get_text(" ", strip=True) if p else ""
    return abstract_text


def get_sections(tag: BeautifulSoup):
    """Recursively extract sections from a BeautifulSoup subtree.

    Returns a list of dicts: {title, paragraphs, subsections}
    """
    if tag is None:
        return []
    section_tags = tag.find_all("section", recursive=False)
    sections = []
    for section_tag in section_tags:
        paragraphs = [p.get_text(" ", strip=True) for p in section_tag.find_all("p")]
        # pick a heading within this section if present
        heading = section_tag.find(["h1","h2","h3","h4","h5","h6","h7"]) 
        title = heading.get_text().strip() if heading else ""
        section_dict = {
            "title": title,
            "paragraphs": "\n".join(paragraphs),
            "subsections": get_sections(section_tag)
        }
        sections.append(section_dict)
    return sections
