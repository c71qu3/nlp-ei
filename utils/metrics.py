
from bert_score import BERTScorer




def calculate_bert_score(candidate:str, reference:str):
        
    scorer = BERTScorer(model_type='bert-base-uncased')
    P, R, F1 = scorer.score([candidate], [reference])
    return P,R, F1


def evaluate_summary(candidate:str, reference:str):
    P, R, F1  = calculate_bert_score(candidate=candidate, reference=reference)
    print(f"Summary reached a bert F1 score of {F1}")