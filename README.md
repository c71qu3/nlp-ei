# Natural Language Processing and Information Extraction
Repository for the NLP and Information Extraction course project (2025 winter semester).

## Project Topic : Retrieval-Augmented Generation for summarization

### Overview
Our team is working on Topic 14: Retrieval-Augmented Generation for Summarization. The goal is to build a RAG-based summarization system that improves the reliability of LLM-generated summarize by hrounding then in relevant chunks of source documents.

### Chosen Domain
We are focusing on academic research papers from the Computer Science - Artificial Intelligence domain, sourced from arXiv. This also helps us gain consistent document structure (abstract, introduction, methods, results, conclusions), also ables us to test summarization across different paper types within AI

---

## Milestone 3

### RAG-Enhanced Summarization

For Milestone 3, we developed a RAG-based summarization system that combines statistical retrieval with LLM reasoning to generate high-quality scientific abstracts. Building on our baseline evaluations, we implemented a hybrid approach using BM25 for sentence selection and structured prompting for coherent summary generation.

#### [Initial Core Implementation: BM25 + LLM](best_method/base_bm25_plus_llm.ipynb)

**1. Document Processing:**
The system breaks papers into sentences, tokenizes and lemmatizes text for improved retrieval accuracy.

**2. BM25 Sentence Selection:**
BM25 scoring ranks sentences by relevance to the full document, selecting the top-N most representative sentences. This extractive approach proved most reliable in our baseline comparisons.

**3. LLM-Based Summary Shaping:**
Selected sentences are passed to an LLM (OpenAI o4-mini) with structured prompts to:
- Reorder sentences for logical flow
- Discard less relevant information
- Generate a concise 5-8 sentence summary
- Maintain objective, third-person academic tone

#### [Section-Based Scaffolded Approach](best_method/base_bm25_plus_llm_sectioned.ipynb)

**1. Section Bucketing:**
Papers are parsed into semantic sections (Introduction, Methods, Results, Conclusion) using regex pattern matching on headers.

**2. Hierarchical Summarization:**
Each section is independently summarized with section-specific prompts:
- Introduction: Problem space and proposed solution
- Methods: Technical approach and implementation
- Results: Quantitative metrics and key findings
- Conclusion: Implications and future work

**3. Summary Assembly:**
Section summaries are combined into a cohesive abstract using a joiner prompt that ensures logical transitions and maintains information density.

**4. Refinement Step:**
A second LLM pass refines the draft to:
- Increase information density
- Preserve all metrics and proper nouns
- Remove filler phrases
- Target 150-250 word count

**5. Fact-Based Evaluation:**
We developed a novel evaluation framework that generates 5 fact-based questions covering problem, methodology, results, dataset, and limitations. Summaries are scored on recall accuracy compared to ground truth answers from the full paper.
The results can be found in [Quantitave Review](data_analysis/quantitative_review.ipynb)

#### [Verbatim RAG Integration](best_method/with_verbatim_rag.ipynb)

We integrated the Verbatim RAG framework for loading and processing documents, grounding LLM-generated summaries with exact citations from source documents, preventing hallucination and enabling verifiable claims.

### Methodology Comparison

The section-based approach (V1_SCAFFOLDED_TEMPLATING and REFINED_SCAFFOLDED_TEMPLATING) outperforms direct BM25 methods by:
- Capturing structured information across all paper sections
- Maintaining better coherence through hierarchical processing
- Preserving critical quantitative results
- Enabling targeted refinement for publication-quality output

### Data Structure
```
best_method/
├── base_bm25_plus_llm.ipynb
├── base_bm25_plus_llm_sectioned.ipynb
├── with_verbatim_rag.ipynb
└── summary_templating.json
```

Implementations can be found in the `best_method/` directory. The `summary_templating.json` file contains generated summaries for evaluation papers.

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

**5. [Chunk-and-Summarize (Two-pass LLM)](baselines/LLMSubsectionSummaryApproach.ipynb):**
The scientific paper is divided into manageable chunks. These are effectively the sections and its subsections (building a tree of different sections levels) Each chunk is summarized independently using an LLM in a bottom up approach.
These intermediate summaries are then concatenated and passed through (bottom-up) the LLM again to produce the final summary. This means that every section has the summary of deeper level subsections as context information.

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