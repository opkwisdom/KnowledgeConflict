import re
import string
from collections import Counter
from typing import List

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    def replace_num(text):
        word_to_number = {
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9"
        }

        pattern = re.compile(r'\b(' + '|'.join(word_to_number.keys()) + r')\b')
        text = pattern.sub(lambda x: word_to_number[x.group()], text)

        return text

    return replace_num(white_space_fix(remove_articles(remove_punc(lower(s)))))

def check_answer(a_pred: str, a_true: List[str]) -> bool:
    """Simple grading function to compare predicted and true answers."""
    return any(normalize_answer(ans) in normalize_answer(a_pred) for ans in a_true)

def recall(a_pred: str, a_true: List[str]) -> float:
    """Compute recall of predicted answer against true answers."""
    pred_tokens = normalize_answer(a_pred).split()
    max_recall = 0.0

    for ans in a_true:
        ans_tokens = normalize_answer(ans).split()
        if len(ans_tokens) == 0:
            continue

        common = Counter(ans_tokens) & Counter(pred_tokens)
        score = sum(common.values()) / len(ans_tokens)
        max_recall = max(max_recall, score)

    return max_recall

def precision(a_pred: str, a_true: List[str]) -> float:
    """Compute precision of predicted answer against true answers."""
    pred_tokens = normalize_answer(a_pred).split()
    total_pred_tokens = len(pred_tokens)
    max_precision = 0.0

    if not pred_tokens:
        return 0.0

    for ans in a_true:
        ans_tokens = normalize_answer(ans).split()
        common = Counter(ans_tokens) & Counter(pred_tokens)
        score = sum(common.values()) / total_pred_tokens

        max_precision = max(max_precision, score)

    return max_precision

def f1_score(a_pred: str, a_true: List[str]) -> float:
    """Compute F1 score of predicted answer against true answers."""
    prec = precision(a_pred, a_true)
    rec = recall(a_pred, a_true)

    if prec + rec == 0:
        return 0.0

    return 2 * (prec * rec) / (prec + rec)

if __name__ == "__main__":
    # Simple test
    pred = "The capital of France is Paris."
    trues = ["Paris", "The capital city is Paris."]

    recall_score = recall(pred, trues)
    precision_score = precision(pred, trues)
    f1 = f1_score(pred, trues)
    import pdb; pdb.set_trace()
    print(check_answer(pred, trues))  # Should return True