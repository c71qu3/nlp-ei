# Natural Language Processing and Information Extraction
Repository for the NLP and Information Extraction course project (2025 winter semester).

## Project Topic : Retrieval-Augmented Generation for summarization

### Overview
Our team is working on Topic 14: Retrieval-Augmented Generation for Summarization. The goal is to build a RAG-based summarization system that improves the reliability of LLM-generated summarize by hrounding then in relevant chunks of source documents.

### Chosen Domain
We are focusing on academic research papers from the Computer Science - Artificial Intelligence domain, sourced from arXiv. This also helps us gain consistent document structure (abstract, introduction, methods, results, conclusions), also ables us to test summarization across different paper types within AI

---

## Milestone 2

### Extractive and Abstractive Summarization Baselines
For Milestone 2, we implemented and evaluated a suite of extractive and abstractive summarization algorithms on a subset of 10 representative papers from our dataset. The purpose is to establish strong baselines and explore the effect of various strategies.

![Baseline work pipeline](reference/NLP%20&%20Information%20Extraction%20assignment%20group%20-%202025-11-29%2012.28.14.jpg)

#### [Rule-based Extractive Methods](baselines/RulesBasedModel.ipynb)

**1. Lead-N:**
We extract the first _N_ sentences from the paper as the summary baseline.

**2. Lead-N by Section:**
Rather than the whole paper, we extract the first _N_ sentences from each section and concatenate these.

#### [Statistical Extractive Methods](baselines/vector_space_models.ipynb)

**3. TF-IDF Summarization:**
We compute the Term Frequency-Inverse Document Frequency (TF-IDF) score for each sentence, ranking all sentences by their TF-IDF scores relative to the entire document, selecting the top _N_ as the summary.

**4. BM25 Summarization:**
Using the BM25 algorithm, each sentence is scored based on its lexical relevance to the paper. Sentences with the highest BM25 scores are selected.

#### LLM-based Abstractive Methods

**5. [Chunk-and-Summarize (Two-pass LLM)](summarization/llm_approach.py):**
The scientific paper is divided into manageable chunks. Each chunk is summarized independently using an LLM.
These intermediate summaries are then concatenated and passed through the LLM again to produce the final summary.

**6. [Embedding-based Chunk Selection + LLM Summarization](baselines/pipeline.py)**
Each chunk and the full document are embedded using a sentence embedding model.
The mean embedding of the overall paper is used as a "summary intent" vector.
Each chunk's embedding is compared to this mean via cosine similarity.
The top _N_ most relevant (highest similarity) chunks are selected and concatenated.
The LLM then summarizes only these selected chunks to produce the output.

### Outputs and Evaluation

#### [Quantitave Review](data_analysis/quantitative_review.ipynb)
Here we compare the performance of all implemented summariation methods using standard evaluation metrics.
We compute ROUGE-1, ROUGE-2, and ROUGE-L scores, which measure n-gram overlap and sequence similarity between generated summaries and reference abstracts.
We also report the BERTScore F1 to capture semantic similarity usin contextual embeddings.
These metrics allow us to objectively benchmark extractive and abstractive baselines.

#### [Qualitative Review](data_analysis/qualitative_review.ipynb)
The qualitative review presents sample summaries from each method side by side, providing human-readable comparison of their output quality.
We highlight notable issues repetitions, over- or under-inclussion of details, factual errors, and incoherence.

### Data Structure
```
baselines/
└── results/
data_analysis/
```

The code to generate the summaries can be found in `baselines`, the output files in `baselines/results/`.
The qualitative and quantitative reviews can be found in the `data_analysis` directory.

## Milestone 1

### Data Collection
We have collected 50 papers from arXiv's cs.AI category. For each paper, we extract and store:
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
data/
├── processed/
└── raw/
    ├── abstracts/
    └── htmls/
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