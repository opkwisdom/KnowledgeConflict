import os
from argparse import ArgumentParser
from typing import List

from utils import (
    load_relevance_dataset, RelevanceQAExample
)


def recall(oracle: List[RelevanceQAExample], target: List[RelevanceQAExample], label_type: str) -> float:
    """Compute recall of predicted answer against true answers."""
    n_total, n_positive = 0, 0
    for o, t in zip(oracle, target):
        o_mapping = o.ctx_relevance.mapping
        t_mapping = t.ctx_relevance.mapping
        
        for o_m, t_m in zip(o_mapping.values(), t_mapping.values()):
            if o_m != label_type:
                continue
            n_total += 1
            if o_m == t_m:
                n_positive += 1
    return n_positive / n_total if n_total > 0 else 0.0

def precision(oracle: List[RelevanceQAExample], target: List[RelevanceQAExample], label_type: str) -> float:
    """Compute precision of predicted answer against true answers."""
    n_total, n_positive = 0, 0
    for o, t in zip(oracle, target):
        o_mapping = o.ctx_relevance.mapping
        t_mapping = t.ctx_relevance.mapping
        
        for o_m, t_m in zip(o_mapping.values(), t_mapping.values()):
            if t_m != label_type:
                continue
            n_total += 1
            if o_m == t_m:
                n_positive += 1

    return n_positive / n_total if n_total > 0 else 0.0

def macro_f1_score(oracle: List[RelevanceQAExample], target: List[RelevanceQAExample]) -> float:
    recall_scores = []
    precision_scores = []
    f1_scores = []
    
    for label_type in ["supportive", "contradictory", "irrelevant"]:
        per_label_recall = recall(oracle, target, label_type)
        per_label_precision = precision(oracle, target, label_type)
        
        recall_scores.append(per_label_recall)
        precision_scores.append(per_label_precision)
        f1_scores.append(2 * (per_label_precision * per_label_recall) / (per_label_precision + per_label_recall + 1e-8))
        print(f"Label: {label_type}, Precision: {per_label_precision:.4f}, Recall: {per_label_recall:.4f}, F1: {f1_scores[-1]:.4f}")

    macro_f1_score = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    print(f"Macro F1 Score: {macro_f1_score:.4f}")
    return macro_f1_score, recall_scores, precision_scores, f1_scores

def main():
    parser = ArgumentParser()
    parser.add_argument("--oracle_path", default="data/train/full_train_data_openai_reasoning.json")
    parser.add_argument("--target_path", required=True)
    args = parser.parse_args()
    
    # Load dataset    
    oracle = load_relevance_dataset(args.oracle_path)
    target = load_relevance_dataset(args.target_path)[:len(oracle)]

    f1_score, recall_scores, precision_scores, f1_scores = macro_f1_score(oracle, target)
    output_path = "test/" + os.path.splitext(os.path.basename(args.target_path))[0] + "_quality.txt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        for label_type, recall, precision, f1 in \
            zip(["supportive", "contradictory", "irrelevant"], recall_scores, precision_scores, f1_scores):
            f.write(f"Label: {label_type}, Recall: {recall:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}\n")
        f.write(f"Macro F1 Score: {f1_score:.4f}\n")


if __name__ == "__main__":
    main()