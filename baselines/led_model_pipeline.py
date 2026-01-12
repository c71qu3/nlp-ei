import json
import os
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from tqdm import tqdm
from typing import List, Dict, Any

def chunk_document(document: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    while start < len(document):
        end = start + chunk_size
        chunks.append(document[start:end])
        start += chunk_size - overlap
    return [chunk for chunk in chunks if chunk.strip()]

def generate_embeddings(chunks: List[str], model) -> np.ndarray:
    return model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)

def calculate_average_embedding(embeddings: np.ndarray, model) -> np.ndarray:
    if embeddings.size == 0:
        return np.zeros(model.get_sentence_embedding_dimension())
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

def extract_text_from_sections(sections: List[Dict[str, Any]]) -> str:
    text = ""
    for section in sections:
        text += section.get('title', '') + "\n"
        text += section.get('paragraphs', '') + "\n"
        if 'subsections' in section:
            text += extract_text_from_sections(section['subsections']) + "\n"
    return text.strip()


def combine_with_existing_results(led_results: List[Dict], eval_results_file: Path, output_file: Path):
    print("\n" + "="*60)
    print("Combining LED results with existing evaluation results...")
    print("="*60)
    
    # Load existing evaluation results
    if not eval_results_file.exists():
        print(f"Warning: Evaluation results file not found at {eval_results_file}")
        print("Saving LED results only...")
        with open(output_file, 'w', encoding='utf8') as f:
            json.dump(led_results, f, indent=2, ensure_ascii=False)
        return
    
    with open(eval_results_file, 'r', encoding='utf8') as f:
        eval_results = json.load(f)
    
    # Create a mapping of paper_id to LED results for quick lookup
    led_results_map = {item['paper_id']: item['led_summary'] for item in led_results}
    
    # Add LED results to evaluation results
    for paper in eval_results:
        paper_id = paper['paper_id']
        if paper_id in led_results_map:
            paper['led'] = led_results_map[paper_id]
            print(f"✓ Added LED results for {paper_id}")
        else:
            print(f"⚠ Warning: No LED results found for {paper_id}")
    
    # Save combined results
    with open(output_file, 'w', encoding='utf8') as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Successfully combined results and saved to {output_file}")
    print(f"Total papers processed: {len(eval_results)}")
    print(f"Papers with LED results: {sum(1 for p in eval_results if 'led' in p)}")


def main():
    # --- 1. Setup Paths and Models ---
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_path = project_root / "data" / "raw"
    results_dir = script_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Output files
    led_only_file = results_dir / "long_context_model_results.json"
    eval_results_file = results_dir / "evaluation_results_final.json"  # Change this to match your file
    combined_output_file = results_dir / "evaluation_results_final_combined.json"

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN environment variable is not set. Please set it with your Hugging Face API key.")
        return

    print("Initializing models and API client...")
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    summarizer_model_name = "pszemraj/led-large-book-summary"
    client = InferenceClient(
        provider="hf-inference",
        api_key=hf_token,
        )
    print("Models and client ready.")

    abstract_files = list((data_path / "abstracts").glob("*.txt"))
    if not abstract_files:
        print(f"No papers found in {data_path / 'abstracts'}. Please run the ingestion script first.")
        return
        
    all_summaries = []
    all_abstracts = []
    final_results = []

    # --- 2. Process Papers and Generate Summaries ---
    temp_files = abstract_files[:3]
    for abstract_file in tqdm(temp_files, desc="Processing Papers & Summarizing"):
        paper_id = abstract_file.stem
        sections_file = data_path / "parsed_sections" / f"{paper_id}.json"

        if not sections_file.exists():
            continue

        with open(abstract_file, "r", encoding="utf8") as f:
            abstract_text = f.read()
        
        with open(sections_file, "r", encoding="utf8") as f:
            document_text = extract_text_from_sections(json.load(f))

        if not document_text.strip():
            continue
        
        semantic_chunks = chunk_document(document_text)
        embeddings = generate_embeddings(semantic_chunks, embedding_model)
        avg_embedding = calculate_average_embedding(embeddings, embedding_model)
        top_chunks = get_top_n_chunks(semantic_chunks, embeddings, avg_embedding, n=5)
        concatenated_chunks = "\n".join(top_chunks)

        # Generate summary with the Inference API
        try:
            response = client.summarization(
                concatenated_chunks,
                model=summarizer_model_name,
            )
            # The client returns a dataclass object, we access the text with .summary_text
            summary_text = response.summary_text
        except Exception as e:
            print(f"API Error for paper {paper_id}: {e}")
            summary_text = "" # Assign empty summary on error
        
        all_summaries.append(summary_text)
        all_abstracts.append(abstract_text)
        final_results.append({"paper_id": paper_id, "led_summary": {"summary": summary_text}})

    # --- 3. Evaluate Summaries ---
    print("\nCalculating evaluation scores...")

    # Cosine Similarity
    print("Calculating cosine similarity scores...")
    summary_embeddings = generate_embeddings(all_summaries, embedding_model)
    abstract_embeddings = generate_embeddings(all_abstracts, embedding_model)
    for i, result in enumerate(final_results):
        sim = compute_similarity(summary_embeddings[i], abstract_embeddings[i])
        result['led_summary']['similarity_to_abstract'] = sim

    # ROUGE Scores
    print("Calculating ROUGE scores...")
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    for i, result in enumerate(tqdm(final_results, desc="ROUGE Scoring")):
        scores = rouge.score(all_abstracts[i], all_summaries[i])
        result['led_summary']['rouge1'] = scores['rouge1'].fmeasure
        result['led_summary']['rouge2'] = scores['rouge2'].fmeasure
        result['led_summary']['rougeL'] = scores['rougeL'].fmeasure

    # BERTScore
    print("Calculating BERTScore...")
    P, R, F1 = bert_score(all_summaries, all_abstracts, lang="en", verbose=True, model_type='distilbert-base-uncased')
    for i, result in enumerate(final_results):
        result['led_summary']['bertscore_f1'] = F1[i].item()

    # --- 4. Save LED-only Results ---
    with open(led_only_file, "w", encoding="utf8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    print(f"\nLED results saved to {led_only_file}")

    # --- 5. Combine with Existing Results ---
    combine_with_existing_results(final_results, eval_results_file, combined_output_file)
    
    print("\n" + "="*60)
    print(f"Pipeline complete!")
    print(f"LED-only results: {led_only_file}")
    print(f"Combined results: {combined_output_file}")
    print("="*60)


if __name__ == '__main__':
    main()