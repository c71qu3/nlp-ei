# Pipeline: arXiv ingestion (Async)

This small package reproduces the notebook ingestion logic as a standalone Python pipeline with **async/await for concurrent downloads** — much faster than the sequential notebook approach!

Files:
- `ingest.py` — CLI/script to fetch recent arXiv papers, extract abstracts and structured sections, and write them under `data/raw/`. Uses `aiohttp` and `asyncio` for parallel downloads.
- `utils.py` — small helpers ported from the notebook.
- `requirements.txt` — minimal dependencies for the pipeline.

## Quick start (from repository root):

```bash
# Basic usage (downloads 50 papers with 10 concurrent requests)
python -m pipeline.ingest --category cs.AI --n-papers 50

# Faster: increase concurrency to 20
python -m pipeline.ingest --category cs.AI --n-papers 100 --concurrency 20

# Debug mode
python -m pipeline.ingest --category cs.AI --n-papers 10 --verbose
```

## Performance

The async version is **significantly faster** than sequential downloads:
- **Sequential** (notebook): ~2-3 seconds per paper → 50 papers = ~2-3 minutes
- **Async** (this pipeline): 10 concurrent downloads → 50 papers = ~20-30 seconds

Adjust `--concurrency` to control the number of simultaneous downloads (default: 10). Be respectful to arXiv servers — don't set it too high!

## Notes
- This pipeline intentionally does not modify any notebook files.
- `pdf_preprocess.ipynb` is intentionally not yet integrated; we'll add it later when the notebook code is finalized.
- The pipeline automatically skips papers that have already been downloaded.
