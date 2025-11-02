# Natural Language Processing and Information Extraction
Repository for the NLP and Information Extraction course project (2025 winter semester).

## Project Topic : Retrieval-Augmented Generation for summarization

### Overview
Our team is working on Topic 14: Retrieval-Augmented Generation for Summarization. The goal is to build a RAG-based summarization system that improves the reliability of LLM-generated summarize by hrounding then in relevant chunks of source documents.

### Chosen Domain
We are focusing on academic research papers from the Computer Science - Artificial Intelligence domain, sourced from arXiv. This also helps us gain consistent document structure (abstract, introduction, methods, results, conclusions), also ables us to test summarization across different paper types within AI

---

## Milestone 1

### Data Collection
We have collected 46 papers from arXiv's cs.AI category. For each paper, we extract and store:
- Full HTML Content
- Abstract text
- Structured sections (parsed into JSON format with hierarchical subsections)

Below is the link to our Dataset

**Dataset**: https://drive.proton.me/urls/VTYYKYJF04#c8bWZDYY82hF

The data collection was initially implemented in Jupyter notebook (`data_ingestion/arxiv_scraper.ipynb`) that:
- Fetches recent papers listing from arXiv
- Downloads each paper's HTML page
- Parses and extract the abstract
- Recursively parses the paper structure into sections and subsections
- Saves 3 files per paper: raw HTMl, abstract text and structured JSON

### PDF Text Extraction
We have also explored PDF text extraction for processing literature papers. The notebook `data_preprocessing/pdf_preprocess.ipynb` implements:
- Text extraction from PDF files using `pdfminer.six`
- Text cleaning (removing hyphenated line breaks, CID markers)
- Abstract extraction using regex patterns
- Tokenization and lemmatization using NLTK
- Output to JSON format with preprocessed sentences

An exploration for potential future use cases involving PDF processing

### Code Organization
The notebook code has been converted into a reusable python pipeline located in the `pipeline/` directory:
- `pipeline/ingest.py` - Main ingest script with async downloads for fast processing
- `pipeline/unils.py` - Helper functions for parsing
- Uses `aiohttp` for concurrent HTTP requests instead of sequential downloads

The pipeline can be run from the command line and allows configuring the number of papers and concurrent awaylable

### Data Structure
```
data/raw/
├── abstracts/          # Plain text abstracts (46 files)
├── htmls/              # Full HTML content (46 files)
└── parsed_sections/    # Structured JSON (46 files)
```

Each JSON file contains sections with titles, paragraphs, and nested subsections:
```json
[
  {
    "title": "1 Introduction",
    "paragraphs": "...",
    "subsections": [...]
  }
]
```

This completes the requirements for Milestone 1

## How to Use

### Setup
Install the required dependencies:
```bash
pip install -r pipeline/requirements.txt


### Running the Pipeline
python -m pipeline.ingest --category cs.AI --n-papers 50

#Adjusting Concurrency
python -m pipeline.ingest --n-papers 100 --concurrency 20

# Debug mode
python -m pipeline-ingest --n-papers 5 -verbose

```

### Accessing the data
- Abstracts: `data/raw/abstracts/*.txt`
- Full HTML: `data/raw/htmls/*.txt`
- Structured sections: `data/raw/parsed/*`