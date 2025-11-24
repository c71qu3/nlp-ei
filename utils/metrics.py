
from bert_score import BERTScorer
from rouge_score import rouge_scorer



def calculate_bert_score(candidate:str, reference:str):
        
    scorer = BERTScorer(model_type='bert-base-uncased')
    P, R, F1 = scorer.score([candidate], [reference])
    return P,R, F1

def calculate_rouge_score(candidate:str, reference:str):

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(candidate,
                        reference)
    return scores


def evaluate_summary(candidate:str, reference:str):
    P, R, F1  = calculate_bert_score(candidate=candidate, reference=reference)
    print(f"Summary reached a bert F1 score of {F1}")

    rouge_scores = calculate_rouge_score(candidate=candidate, reference=reference)
    print(f"Summary reached a rouge score of {rouge_scores}")