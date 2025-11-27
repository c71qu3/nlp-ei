import asyncio
import os
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from pathlib import Path

from dotenv import load_dotenv

from gemini_service import summarize_with_gemini
from groq_service import summarize_with_groq

# Load environment variables from .env file
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Load the sentence transformer model
# This will download the model on first run
print("Loading sentence-transformer model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")


def chunk_document(document: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Splits the document into chunks of a specified size with overlap.
    """
    chunks = []
    start = 0
    while start < len(document):
        end = start + chunk_size
        chunks.append(document[start:end])
        start += chunk_size - overlap
    return [chunk for chunk in chunks if chunk.strip()]

def generate_embeddings(chunks: List[str]) -> np.ndarray:
    """
    Generates embeddings for each chunk using a sentence-transformer model.
    """
    print(f"Generating embeddings for {len(chunks)} chunks...")
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    return embeddings

def calculate_average_embedding(embeddings: np.ndarray) -> np.ndarray:
    """
    Calculates the average meaning embedding.
    """
    if embeddings.size == 0:
        return np.zeros(embedding_model.get_sentence_embedding_dimension())
    avg_embedding = np.mean(embeddings, axis=0)
    return avg_embedding

def compute_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Computes cosine similarity between two embeddings.
    """
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    cosine_sim = np.dot(embedding1, embedding2) / (norm1 * norm2)
    return float(cosine_sim)


def get_top_n_chunks(chunks: List[str], embeddings: np.ndarray, avg_embedding: np.ndarray, n: int) -> List[str]:
    """
    Selects the top N chunks based on their similarity to the average embedding.
    """
    similarities = [compute_similarity(avg_embedding, emb) for emb in embeddings]
    
    sorted_chunks_with_scores = sorted(zip(chunks, similarities), key=lambda item: item[1], reverse=True)
    
    top_chunks = [chunk for chunk, score in sorted_chunks_with_scores[:n]]
    
    return top_chunks


async def run_summarization_pipeline(document: str, abstract: str, top_n: int = 5) -> dict:
    """
    Runs the full summarization pipeline.
    """
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
    """Recursively extracts text from the parsed sections JSON."""
    text = ""
    for section in sections:
        text += section.get('title', '') + "\n"
        text += section.get('paragraphs', '') + "\n"
        if 'subsections' in section:
            text += extract_text_from_sections(section['subsections']) + "\n"
    return text.strip()


async def main():
    """
    Main function to load actual paper data and run the pipeline on them.
    """
    # Build a path to the data directory relative to this script's location
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_path = project_root / "data" / "raw"

    abstract_files = list((data_path / "abstracts").glob("*.txt"))
    
    if not abstract_files:
        print(f"No papers found in {data_path / 'abstracts'}. Please run the ingestion script first.")
        return
        
    all_results = []
    print(f"Found {len(abstract_files)} papers. Processing all of them...")
    
    # test_files = abstract_files[:3]
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

    # Save all results to a JSON file
    results_dir = project_root / "baselines/results"
    results_dir.mkdir(exist_ok=True)
    output_file = results_dir / "baseline_results.json"

    with open(output_file, "w", encoding="utf8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
        
    print(f"\nPipeline complete. All results saved to {output_file}")


if __name__ == '__main__':
    if not GEMINI_API_KEY or not GROQ_API_KEY:
        print("Please set GEMINI_API_KEY and GROQ_API_KEY in a .env file.")
    else:
        asyncio.run(main())