import json
from dataclasses import dataclass
from typing import List, Dict
import os


@dataclass
class InferenceResult:
    id: int
    question: str
    pred_answer: str
    answers: List[str]
    is_correct: bool


def load_result_json(path: str):
    with open(path, 'r') as f:
        return json.load(f)

def get_summary(
    results: List[InferenceResult],
    id_dict: Dict[int, str],
    output_path: str = None,
) -> Dict[str, float]:
    summary = {key: [] for key in id_dict.values()}

    for i, res in enumerate(results):
        res_type = id_dict[i]
        summary[res_type].append(res["is_correct"])

    filtered_summary = {}
    for k, v in summary.items():
        total = len(v)
        correct = sum(v)
        accuracy = correct / total if total > 0 else 0.0
        filtered_summary[k] = {
            "total": total,
            "correct": correct,
            "accuracy": round(accuracy, 4),
        }

    if output_path is not None:
        with open(output_path, 'w') as f:
            json.dump(filtered_summary, f, ensure_ascii=False, indent=4)

    return filtered_summary

# Load results
reference_result_path = "../results/main/e2e_openai/ratio=0.3_prompt=base/20251204_054423/inference_results.json"
pure_result_path = "../results/pure/prompt=pure-llm/20251203_114736/inference_results.json"
rag_result_path = "../results/rag/prompt=base/20251203_111619/inference_results.json"

reference_results = load_result_json(reference_result_path)
pure_results = load_result_json(pure_result_path)
rag_results = load_result_json(rag_result_path)

total_keys = ["param_true", "param_positive", "param_negative", "param_irrelevant", "param_multiple"]
id_dict = {}

for key in total_keys:
    for result in reference_results[key]:
        id_dict[result["id"]] = key


os.makedirs("analysis_results", exist_ok=True)
# pure results analysis
pure_results = pure_results["pure_result"]
pure_summary = get_summary(pure_results, id_dict, output_path="analysis_results/pure_summary.json")

# rag results analysis
rag_results = rag_results["rag_result"]
rag_summary = get_summary(rag_results, id_dict, output_path="analysis_results/rag_summary.json")