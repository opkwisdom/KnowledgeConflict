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

    for res in results:
        res_id = res.get("id")
        if res_id not in id_dict:
            continue
        res_type = id_dict[res_id]
        if "metrics" in res:
            summary[res_type].append(res["metrics"])

    filtered_summary = {}
    for k, v in summary.items():
        total = len(v)

        correct = sum([1 for res in v if res["soft_em"]])
        recall = sum([res["recall"] for res in v]) / total if total > 0 else 0.0
        precision = sum([res["precision"] for res in v]) / total if total > 0 else 0.0
        f1 = sum([res["f1"] for res in v]) / total if total > 0 else 0.0
        
        accuracy = correct / total if total > 0 else 0.0
        filtered_summary[k] = {
            "total": total,
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
        }

    if output_path is not None:
        with open(output_path, 'w') as f:
            json.dump(filtered_summary, f, ensure_ascii=False, indent=4)

    return filtered_summary

def analyze_param_true_diff(
    target_results: List[Dict], 
    ref_results_dict: Dict[int, Dict], 
    id_map: Dict[int, str],
    target_name: str = "Pure",
    output_file: str = "analysis_results/param_true_diff.json"
):
    """
    param_true 카테고리에서 Target 모델과 Reference 모델의 정답 여부가 다른 경우를 추출
    """
    diff_cases = []
    
    for res in target_results:
        res_id = res.get("id")
        
        # 1. param_true 인지 확인
        if id_map.get(res_id) != "param_true":
            continue
            
        # 2. Reference 결과 가져오기
        ref_res = ref_results_dict.get(res_id)
        if not ref_res:
            continue
            
        # 3. 정답 여부 비교 (soft_em 기준)
        # metrics가 딕셔너리로 들어있다고 가정
        target_correct = res["metrics"].get("soft_em", False)
        ref_correct = ref_res["metrics"].get("soft_em", False)
        
        # 4. 결과가 다르면(하나만 맞고 하나는 틀림) 저장
        if target_correct != ref_correct:
            diff_cases.append({
                "id": res_id,
                "question": res["question"],
                "gold_answers": res["answers"],
                f"{target_name}_pred": res["pred_answer"],
                f"{target_name}_correct": target_correct,
                "Reference_pred": ref_res["pred_answer"],
                "Reference_correct": ref_correct,
            })
            
    # 결과 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(diff_cases, f, indent=4, ensure_ascii=False)
        
    print(f"[{target_name} vs Reference] param_true 차이 분석 완료: {len(diff_cases)}건 발견 -> {output_file}")

# Load results
reference_result_path = "../results/main/e2e_openai/ratio=0.3_prompt=base/20251206_052138/inference_results.json"
pure_result_path = "../results/pure/prompt=pure-llm-brief-2/20251205_133717/inference_results.json"
rag_result_path = "../results/rag/prompt=base/20251204_144609/inference_results.json"

reference_results = load_result_json(reference_result_path)
pure_results = load_result_json(pure_result_path)
rag_results = load_result_json(rag_result_path)

ref_results_dict = {}
total_keys = ["param_true", "param_positive", "param_negative", "param_irrelevant", "param_multiple"]
id_dict = {}

for key in total_keys:
    for result in reference_results[key]:
        id_dict[result["id"]] = key
        ref_results_dict[result["id"]] = result


os.makedirs("analysis_results", exist_ok=True)
# pure results analysis
pure_results = pure_results["pure_result"]
pure_summary = get_summary(pure_results, id_dict, output_path="analysis_results/pure_summary.json")

# rag results analysis
rag_results = rag_results["rag_result"]
rag_summary = get_summary(rag_results, id_dict, output_path="analysis_results/rag_summary.json")

analyze_param_true_diff(
    target_results=pure_results,
    ref_results_dict=ref_results_dict,
    id_map=id_dict,
    target_name="Pure",
    output_file="analysis_results/pure_vs_ours_param_true_diff.json"
)