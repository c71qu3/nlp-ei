
from bert_score import BERTScorer
from rouge_score import rouge_scorer



def calculate_bert_score(candidate:str, reference:str):
        
    scorer = BERTScorer(model_type='bert-base-uncased')
    P, R, F1 = scorer.score([candidate], [reference])
    return {
        "bertscore_precision": float(P[0]),
        "bertscore_recall": float(R[0]),
        "bertscore_f1": float(F1[0])
    }

def calculate_rouge_score(candidate:str, reference:str):

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(candidate,
                        reference)
    result = {}
    for score_name, score_v in scores.items():
        result[f"{score_name}_precision"] = score_v.precision
        result[f"{score_name}_recall"] = score_v.recall
        result[f"{score_name}_fmeasure"] = score_v.fmeasure
    return result


def evaluate_summary(candidate:str, reference:str):

    bert_scores  = calculate_bert_score(candidate=candidate, reference=reference)
    rouge_scores = calculate_rouge_score(candidate=candidate, reference=reference)
    result = {}

    result.update(bert_scores)
    result.update(rouge_scores)
    return result