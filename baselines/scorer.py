from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util


def get_rouge_scores(abstract: str, lead_n: str):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(abstract, lead_n)
    
    return scores


def get_embedding_cosine_similarity_score(model: SentenceTransformer, abstract: str, summary: str) -> float:
    emb1 = model.encode(abstract, convert_to_tensor=True)
    emb2 = model.encode(summary, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()

