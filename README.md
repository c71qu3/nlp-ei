# Natural Language Processing and Information Extraction
Repository for the NLP and Information Extraction course project (2025 winter semester).

## Project Topic: Retrieval-Augmented Generation for Summarization

### Overview
Our team is working on Topic 14: Retrieval-Augmented Generation for Summarization. The goal is to build a RAG-based summarization system that improves the reliability of LLM-generated summaries by grounding them in relevant chunks of source documents.

### Chosen Domain
We are focusing on academic research papers from the Computer Science - Artificial Intelligence domain, sourced from arXiv. This domain provides:
- Consistent document structure (abstract, introduction, methods, results, conclusions)
- Technical vocabulary and concepts
- Ability to test summarization across different paper types within AI

---

## Work Completed So Far

### Data Collection
We have collected 46 papers from arXiv's cs.AI category. For each paper, we extract and store:
- Full HTML content
- Abstract text
- Structured sections (parsed into JSON format with hierarchical subsections)

The data collection was initially implemented in a Jupyter notebook (`data_ingestion/arxiv_scraper.ipynb`) that:
- Fetches the recent papers listing from arXiv
- Downloads each paper's HTML page
- Parses and extracts the abstract
- Recursively parses the paper structure into sections and subsections
- Saves three files per paper: raw HTML, abstract text, and structured JSON

### PDF Text Extraction
We have also explored PDF text extraction for processing literature papers. The notebook `data_preprocessing/pdf_preprocess.ipynb` implements:
- Text extraction from PDF files using `pdfminer.six`
- Text cleaning (removing hyphenated line breaks, CID markers)
- Abstract extraction using regex patterns
- Tokenization and lemmatization using NLTK
- Output to JSON format with preprocessed sentences

This work is separate from the main arXiv data collection and was done as exploration for potential future use cases involving PDF processing.

### Code Organization
The notebook code has been converted into a reusable Python pipeline located in the `pipeline/` directory:
- `pipeline/ingest.py` - Main ingestion script with async downloads for faster processing
- `pipeline/utils.py` - Helper functions for parsing
- Uses `aiohttp` for concurrent HTTP requests instead of sequential downloads

The pipeline can be run from the command line and allows configuring the number of papers and concurrent downloads.

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

This completes the requirements for Milestone 1 (data collection and preprocessing in standard format).

---

## How to Use

### Setup
Install the required dependencies:
```bash
pip install -r pipeline/requirements.txt
```

### Running the Pipeline
To collect papers from arXiv:
```bash
# Collect 50 papers with default settings
python -m pipeline.ingest --category cs.AI --n-papers 50

# Adjust concurrency for faster/slower downloads
python -m pipeline.ingest --n-papers 100 --concurrency 20

# Debug mode
python -m pipeline.ingest --n-papers 5 --verbose
```

The pipeline will create the necessary directories and save files to `data/raw/`.

### Accessing the Data
- Abstracts: `data/raw/abstracts/*.txt`
- Full HTML: `data/raw/htmls/*.txt`
- Structured sections: `data/raw/parsed_sections/*.json`

### Using the Notebooks
The original notebooks can be used for exploratory work:
- `data_ingestion/arxiv_scraper.ipynb` - For arXiv HTML scraping
- `data_preprocessing/pdf_preprocess.ipynb` - For PDF text extraction and preprocessing

---

## Repository Structure
```
nlp-ei/
├── data/
│   └── raw/
│       ├── abstracts/
│       ├── htmls/
│       └── parsed_sections/
├── data_ingestion/
│   ├── arxiv_scraper.ipynb
│   └── requirements.txt
├── data_preprocessing/
│   └── pdf_preprocess.ipynb
├── pipeline/
│   ├── ingest.py
│   ├── utils.py
│   ├── requirements.txt
│   └── README.md
└── README.md
```

---

## References
- Bhandari, M., et al. (2020). "Re-evaluating Evaluation in Text Summarization." EMNLP 2020.
- arXiv Computer Science - AI: https://arxiv.org/list/cs.AI/recent



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
 \Structured sections: `data/raw/parsed/*`