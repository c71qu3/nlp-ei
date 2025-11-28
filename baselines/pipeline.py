import asyncio
import os
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from pathlib import Path
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from tqdm import tqdm
from collections import OrderedDict

from dotenv import load_dotenv

from gemini_service import summarize_with_gemini
from groq_service import summarize_with_groq

# Load environment variables from .env file
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Load the sentence transformer model
print("Loading sentence-transformer model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")


def chunk_document(document: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    while start < len(document):
        end = start + chunk_size
        chunks.append(document[start:end])
        start += chunk_size - overlap
    return [chunk for chunk in chunks if chunk.strip()]

def generate_embeddings(chunks: List[str]) -> np.ndarray:
    print(f"Generating embeddings for {len(chunks)} chunks...")
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    return embeddings

def calculate_average_embedding(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.size == 0:
        return np.zeros(embedding_model.get_sentence_embedding_dimension())
    avg_embedding = np.mean(embeddings, axis=0)
    return avg_embedding

def compute_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    cosine_sim = np.dot(embedding1, embedding2) / (norm1 * norm2)
    return float(cosine_sim)


def get_top_n_chunks(chunks: List[str], embeddings: np.ndarray, avg_embedding: np.ndarray, n: int) -> List[str]:
    similarities = [compute_similarity(avg_embedding, emb) for emb in embeddings]
    
    sorted_chunks_with_scores = sorted(zip(chunks, similarities), key=lambda item: item[1], reverse=True)
    
    top_chunks = [chunk for chunk, score in sorted_chunks_with_scores[:n]]
    
    return top_chunks


async def run_summarization_pipeline(document: str, abstract: str, top_n: int = 5) -> dict:
    print("1. Chunking document...")
    chunks = chunk_document(document)

    print("2. Generating embeddings...")
    embeddings = generate_embeddings(chunks)

    print("3. Calculating average embedding...")
    avg_embedding = calculate_average_embedding(embeddings)

    print(f"4. Selecting top {top_n} chunks...")
    top_chunks = get_top_n_chunks(chunks, embeddings, avg_embedding, top_n)
    concatenated_chunks = "\n".join(top_chunks)

    print("5. Summarizing with Gemini and Groq...")
    gemini_task = summarize_with_gemini(concatenated_chunks, GEMINI_API_KEY)
    groq_task = summarize_with_groq(concatenated_chunks, GROQ_API_KEY)

    gemini_summary, groq_summary = await asyncio.gather(gemini_task, groq_task)

    print("6. Measuring similarity to abstract...")
    # For similarity, we need embeddings for the summaries and the abstract
    summary_and_abstract_texts = [gemini_summary, groq_summary, abstract]
    summary_embeddings = generate_embeddings(summary_and_abstract_texts)
    
    gemini_similarity = compute_similarity(summary_embeddings[0], summary_embeddings[2])
    groq_similarity = compute_similarity(summary_embeddings[1], summary_embeddings[2])

    return {
        "gemini_summary": gemini_summary,
        "groq_summary": groq_summary,
        "gemini_similarity_to_abstract": gemini_similarity,
        "groq_similarity_to_abstract": groq_similarity,
    }

def extract_text_from_sections(sections: List[Dict[str, Any]]) -> str:
    text = ""
    for section in sections:
        text += section.get('title', '') + "\n"
        text += section.get('paragraphs', '') + "\n"
        if 'subsections' in section:
            text += extract_text_from_sections(section['subsections']) + "\n"
    return text.strip()


def calculate_evaluation_scores(all_results: List[Dict], abstracts_dir: Path) -> List[Dict]:
    # Preparing lists for batch BERTScore calculation and for the new structure
    all_gemini_summaries = []
    all_groq_summaries = []
    all_abstracts = []
    nested_results = []

    print("\nPreparing data for scoring and restructuring...")
    for result in all_results:
        paper_id = result['paper_id']
        abstract_file = abstracts_dir / f"{paper_id}.txt"

        if not abstract_file.exists():
            print(f"Warning: Abstract for {paper_id} not found. Skipping.")
            continue
        
        with open(abstract_file, 'r', encoding='utf8') as f:
            abstract_text = f.read()

        # Store data needed for batch processing
        all_gemini_summaries.append(result['gemini_summary'])
        all_groq_summaries.append(result['groq_summary'])
        all_abstracts.append(abstract_text)
        
        # Create the new nested structure
        restructured_item = {
            "paper_id": paper_id,
            "gemini-2.5-flash": {
                "summary": result['gemini_summary'],
                "similarity_to_abstract": result.get('gemini_similarity_to_abstract')
            },
            "llama-3.1-8b-instant": {
                "summary": result['groq_summary'],
                "similarity_to_abstract": result.get('groq_similarity_to_abstract')
            }
        }
        nested_results.append(restructured_item)

    # Calculate ROUGE scores
    print("Calculating ROUGE scores...")
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    for i, result in enumerate(tqdm(nested_results, desc="ROUGE Scoring")):
        abstract = all_abstracts[i]
        gemini_summary = result['gemini-2.5-flash']['summary']
        groq_summary = result['llama-3.1-8b-instant']['summary']

        # Gemini ROUGE scores
        gemini_rouge = scorer.score(abstract, gemini_summary)
        result['gemini-2.5-flash']['rouge1'] = gemini_rouge['rouge1'].fmeasure
        result['gemini-2.5-flash']['rouge2'] = gemini_rouge['rouge2'].fmeasure
        result['gemini-2.5-flash']['rougeL'] = gemini_rouge['rougeL'].fmeasure

        # Groq ROUGE scores
        groq_rouge = scorer.score(abstract, groq_summary)
        result['llama-3.1-8b-instant']['rouge1'] = groq_rouge['rouge1'].fmeasure
        result['llama-3.1-8b-instant']['rouge2'] = groq_rouge['rouge2'].fmeasure
        result['llama-3.1-8b-instant']['rougeL'] = groq_rouge['rougeL'].fmeasure
        
    # Calculate BERTScore in a batch for efficiency
    print("Calculating BERTScore (this may take a while and download a model on first run)...")
    
    # Gemini BERTScore
    P_gemini, R_gemini, F1_gemini = bert_score(all_gemini_summaries, all_abstracts, lang="en", verbose=True, model_type='distilbert-base-uncased')
    for i, result in enumerate(nested_results):
        result['gemini-2.5-flash']['bertscore_f1'] = F1_gemini[i].item()

    # Groq BERTScore
    P_groq, R_groq, F1_groq = bert_score(all_groq_summaries, all_abstracts, lang="en", verbose=True, model_type='distilbert-base-uncased')
    for i, result in enumerate(nested_results):
        result['llama-3.1-8b-instant']['bertscore_f1'] = F1_groq[i].item()

    return nested_results


def reorder_paper_keys(paper_obj):
    """Reorder keys so abstract comes right after paper_id."""
    ordered = OrderedDict()
    
    if 'paper_id' in paper_obj:
        ordered['paper_id'] = paper_obj['paper_id']
    
    if 'abstract' in paper_obj:
        ordered['abstract'] = paper_obj['abstract']
    
    for key in paper_obj:
        if key not in ['paper_id', 'abstract']:
            ordered[key] = paper_obj[key]
    
    return ordered


def add_abstracts_and_reorder(results: List[Dict], abstracts_dir: Path) -> List[Dict]:
    print("\n" + "="*60)
    print("Adding abstracts and reordering keys...")
    print("="*60)
    
    reordered_results = []
    for i, paper_result in enumerate(results):
        paper_id = paper_result['paper_id']
        print(f"[{i+1}/{len(results)}] Adding abstract for {paper_id}")
        
        # Load abstract
        abstract_file = abstracts_dir / f"{paper_id}.txt"
        if abstract_file.exists():
            with open(abstract_file, 'r', encoding='utf8') as f:
                abstract = f.read().strip()
            paper_result['abstract'] = abstract
        else:
            print(f"Warning: Abstract file not found for {paper_id}")
            paper_result['abstract'] = ""
        
        # Reorder keys
        reordered_paper = reorder_paper_keys(paper_result)
        reordered_results.append(reordered_paper)
    
    return reordered_results


async def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_path = project_root / "data" / "raw"

    abstract_files = list((data_path / "abstracts").glob("*.txt"))
    
    if not abstract_files:
        print(f"No papers found in {data_path / 'abstracts'}. Please run the ingestion script first.")
        return
        
    all_results = []
    print(f"Found {len(abstract_files)} papers. Processing all of them...")

    # temp_files = abstract_files[:3]
    for i, abstract_file in enumerate(abstract_files):
        paper_id = abstract_file.stem
        sections_file = data_path / "parsed_sections" / f"{paper_id}.json"

        if not sections_file.exists():
            print(f"Skipping paper {paper_id}: parsed sections file not found.")
            continue

        print(f"\n--- Processing Paper {i+1}/{len(abstract_files)}: {paper_id} ---")

        with open(abstract_file, "r", encoding="utf8") as f:
            abstract_text = f.read()
        
        with open(sections_file, "r", encoding="utf8") as f:
            sections_json = json.load(f)
            document_text = extract_text_from_sections(sections_json)

        if not document_text.strip():
            print(f"Skipping paper {paper_id}: no text found in parsed sections.")
            continue

        results = await run_summarization_pipeline(document_text, abstract_text)
        results['paper_id'] = paper_id
        all_results.append(results)
        
        print(f"\n--- Finished {paper_id} ---")

    # Calculate evaluation scores (ROUGE and BERTScore)
    print("\n" + "="*60)
    print("Starting evaluation scoring...")
    print("="*60)
    abstracts_path = data_path / "abstracts"
    evaluated_results = calculate_evaluation_scores(all_results, abstracts_path)

    # Add abstracts and reorder keys
    final_results = add_abstracts_and_reorder(evaluated_results, abstracts_path)

    # Save final evaluated results to a JSON file
    results_dir = project_root / "baselines/results"
    results_dir.mkdir(exist_ok=True)
    output_file = results_dir / "evaluation_results_final.json"

    with open(output_file, "w", encoding="utf8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
        
    print(f"\n" + "="*60)
    print(f"Pipeline complete! Final evaluation results with abstracts saved to {output_file}")
    print("="*60)


if __name__ == '__main__':
    if not GEMINI_API_KEY or not GROQ_API_KEY:
        print("Please set GEMINI_API_KEY and GROQ_API_KEY in a .env file.")
    else:
        asyncio.run(main())