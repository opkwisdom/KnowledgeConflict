import re
import string
import regex
from collections import Counter
from typing import Dict, List, Union, Tuple
from dataclasses import dataclass, asdict
import logging
import json
import numpy as np


@dataclass
class MetricResult:
    soft_em: bool
    recall: float
    precision: float
    f1: float

@dataclass
class InferenceResult:
    id: int
    question: str
    pred_answer: str
    answers: List[str]
    metrics: MetricResult

class SimpleTokenizer(object):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens


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


def has_answer(a_pred: str, a_true: List[str], tokenizer=SimpleTokenizer()) -> bool:
    """Check if a document contains an answer string."""
    a_pred = normalize_answer(a_pred)
    pred_tokens = tokenizer.tokenize(a_pred, uncased=True)

    if isinstance(a_true, dict):
        a_true = a_true["aliases"]  # TriviaQA

    for ans in a_true:
        ans = normalize_answer(ans)
        ans = tokenizer.tokenize(ans, uncased=True)
        for i in range(0, len(pred_tokens) - len(ans) + 1):
            if ans == pred_tokens[i: i + len(ans)]:
                return True
    return False


# def check_answer(a_pred: str, a_true: List[str]) -> bool:
#     """Simple grading function to compare predicted and true answers."""
#     return any(normalize_answer(ans) in normalize_answer(a_pred) for ans in a_true)

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

### Return all metrics
def compute_metrics(a_pred: str, a_true: Union[str, List[str]]) -> MetricResult:
    """Compute all metrics: soft EM, recall, precision, F1."""
    if isinstance(a_true, str):
        a_true = [a_true]
    
    soft_em = has_answer(a_pred, a_true)
    rec = recall(a_pred, a_true)
    prec = precision(a_pred, a_true)
    f1 = f1_score(a_pred, a_true)

    return MetricResult(
        soft_em=soft_em,
        recall=rec,
        precision=prec,
        f1=f1,
    )


def validate_and_save_results(
    inference_list: Dict[str, List[InferenceResult]],
    output_dir: str,
    logger: logging.Logger,
) -> None:
    summary_path = f"{output_dir}/inference_summary.txt"
    all_results_path = f"{output_dir}/inference_results.json"

    total = len(inference_list)
    correct = sum([1 for res in inference_list if res.metrics.soft_em])
    recall = sum([res.metrics.recall for res in inference_list]) / total if total > 0 else 0.0
    precision = sum([res.metrics.precision for res in inference_list]) / total if total > 0 else 0.0
    f1 = sum([res.metrics.f1 for res in inference_list]) / total if total > 0 else 0.0

    accuracy = correct / total if total > 0 else 0.0
    logger.info(f"Total={total}, Correct={correct}, Accuracy={accuracy:.4f},"
                f" Recall={recall:.4f}, Precision={precision:.4f}, F1={f1:.4f}")
    summary = {
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)
    logger.info(f"Saved inference summary to {summary_path}")
    with open(all_results_path, 'w') as f:
        json_results = [asdict(res) for res in inference_list]
        json.dump(json_results, f, ensure_ascii=False, indent=4)

def compute_ndcg(
    preds: np.ndarray,
    labels: np.ndarray,
    kvalues: List[int] = [1, 3, 5, 10]
) -> List[float]:
    min_labels = np.min(labels)
    if min_labels < 0:
        labels = labels - min_labels

    preds_ranks = np.argsort(-preds)
    ideal_ranks = np.argsort(-labels)

    ndcg_scores = []
    for k in kvalues:
        dcg = np.sum((labels[preds_ranks[:k]] / np.log2(np.arange(2, k + 2))))
        idcg = np.sum((labels[ideal_ranks[:k]] / np.log2(np.arange(2, k + 2))))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)
    return ndcg_scores


if __name__ == "__main__":
    # Simple test
    # pred = "The capital of France is Paris."
    # trues = ["Paris", "The capital city is Paris."]

    # recall_score = recall(pred, trues)
    # precision_score = precision(pred, trues)
    # f1 = f1_score(pred, trues)
    # import pdb; pdb.set_trace()
    # print(has_answer(pred, trues))  # Should return True

    x = np.array([0.5, 0.8, 0.2, 0.7, -0.5, 0.1, 0.1, 0.0, 0.9, 1.5])
    y = np.array([0.3, -0.2, 0.1, 0.6, -0.4, 0.2, 0.0, 0.1, 0.8, 1.0])
    metrics = compute_ndcg(x, y)
    print(metrics)