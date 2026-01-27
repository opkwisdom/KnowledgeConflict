from typing import List, Union
from omegaconf import OmegaConf, DictConfig
from datetime import datetime
from pydantic import BaseModel
from tqdm import tqdm
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

from KVzip.model import ModelKVzip
from kfc_model import KnowledgeFusionCore
from prompt import ALL_PROMPTS, PSEUDO_PASSAGE_PROMPT, GENERATE_PROMPT
from utils import (
    setup_logger, load_config, load_relevance_dataset, load_qa_dataset, compute_metrics,
    MetricResult, RelevanceQAExample, BoostedProbResult
)


class InferenceResult(BaseModel):
    id: int
    question: str
    pred_answer: str
    answers: List[str]
    metrics: MetricResult
    boosted_prob_result: Union[BoostedProbResult, float]


def run_inference(
    config: DictConfig,
    kfc: KnowledgeFusionCore,
    data: List[RelevanceQAExample],
    inspect_mode: bool = True,
    logger = None,
) -> List[InferenceResult]:
    results = []
    for idx, item in tqdm(enumerate(data), desc="Running KFC Inference (No Judge)", total=len(data)):
        question = item.question
        answers = item.answers
        if isinstance(answers, dict):
            answers = answers.get("aliases", None)  # TriviaQA format
        
        a_internal = kfc.generate_internal_answer(question,
                                                  return_probs=config.uncertainty_estimator.return_probs,
                                                  inspect_mode=inspect_mode)
        
        boosted_prob_result = None
        if isinstance(a_internal, tuple):
            a_internal, boosted_prob_result = a_internal

        metrics=compute_metrics(a_internal, answers)
        is_correct = metrics.soft_em

        sample_result = InferenceResult(
            id=idx,
            question=item.question,
            pred_answer=a_internal,
            answers=answers,
            metrics=metrics,
            boosted_prob_result=boosted_prob_result
        )
        results.append(sample_result)
    logger.info("Inference completed.")

    return results

def analyze_auroc(json_path, output_path, logger):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    y_true = []
    y_scores = []
    
    for item in data:
        is_correct = 1 if item['metrics']['soft_em'] else 0
        confidence_score = np.mean(item['boosted_prob_result']['accum_prob'])
        
        y_true.append(is_correct)
        y_scores.append(confidence_score)
            
    if len(set(y_true)) < 2:
        logger.error("Error: 데이터에 정답만 있거나 오답만 있어서 비교 불가능합니다.")
        return

    auroc = roc_auc_score(y_true, y_scores)
    logger.info(f"Final AUROC Score: {auroc:.4f}")
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auroc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # 랜덤 기준선
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Uncertainty Estimation Performance (AUROC)')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(output_path)

# def validate_and_save_results(
#     results: Dict[str, List[InferenceResult]],
#     output_dir: str,
#     logger: logging.Logger,
# ) -> None:
#     summary_path = f"{output_dir}/inference_summary.txt"
#     all_results_path = f"{output_dir}/inference_results.json"
#     summary = {}

#     for k, inference_list in results.items():
#         total = len(inference_list)

#         correct = sum([1 for res in inference_list if res.metrics.soft_em])
#         recall = sum([res.metrics.recall for res in inference_list]) / total if total > 0 else 0.0
#         precision = sum([res.metrics.precision for res in inference_list]) / total if total > 0 else 0.0
#         f1 = sum([res.metrics.f1 for res in inference_list]) / total if total > 0 else 0.0

#         accuracy = correct / total if total > 0 else 0.0
#         logger.info(f"Case {k}: Total={total}, Correct={correct}, Accuracy={accuracy:.4f},"
#                     f" Recall={recall:.4f}, Precision={precision:.4f}, F1={f1:.4f}")
#         summary[k] = {
#             "total": total,
#             "correct": correct,
#             "accuracy": round(accuracy, 4),
#             "recall": round(recall, 4),
#             "precision": round(precision, 4),
#             "f1": round(f1, 4),
#         }
    
#     with open(summary_path, 'w') as f:
#         json.dump(summary, f, ensure_ascii=False, indent=4)
#     logger.info(f"Saved inference summary to {summary_path}")
#     with open(all_results_path, 'w') as f:
#         json_results = {
#             k: [res.model_dump() for res in v] for k, v in results.items()
#         }
#         json.dump(json_results, f, ensure_ascii=False, indent=4)

def main():
    config = load_config()
    cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    expriment_name = f"top_k={config.uncertainty_estimator.top_k}_prompt={config.generate_prompt_name}"
    config.output_dir = os.path.join(config.output_dir, config.data.name, "calibration", expriment_name, cur_time)
    logger = setup_logger(f"main_{cur_time}", config.output_dir)
    logger.info("Configuration Loaded:")
    logger.info(OmegaConf.to_yaml(config))

    repeat_prompt = None
    generate_prompt = None

    if config.task != "pseudo-passage":
        repeat_prompt = ALL_PROMPTS[config.self_task_prompt_name]
    else:
        pair_prompt = PSEUDO_PASSAGE_PROMPT[config.self_task_prompt_name]
        repeat_prompt = pair_prompt["repeat"]
    base_prompt = GENERATE_PROMPT["pure-llm-brief-2"]
    generate_prompt = GENERATE_PROMPT[config.generate_prompt_name]

    logger.info(f"Using base prompt: {base_prompt}")

    # Load data
    if config.data.name == "nq":
        data = load_relevance_dataset(config.data.data_path)
    else:
        data = load_qa_dataset(config.data.data_path)
    
    # data = data[:500]
    logger.info(f"Loaded {len(data)} data entries from {config.data.data_path}")

    # Initialize model
    kvzip = ModelKVzip(config.model.model_name, gen_kwargs=config.model.gen_kwargs, prompt=repeat_prompt)
    logger.info(f"Model {config.model.model_name} initialized.")
    kfc = KnowledgeFusionCore(config, kvzip, generate_prompt, base_prompt, logger)
    logger.info("Knowledge Fusion Core initialized.")

    inference_result = run_inference(config, kfc, data, config.uncertainty_estimator.inspect_mode, logger)
    with open(f"{config.output_dir}/inference_results.json", 'w') as f:
        json_results = [res.model_dump() for res in inference_result]
        json.dump(json_results, f, ensure_ascii=False, indent=4)
    logger.info(f"Saved inference results to {config.output_dir}/inference_results.json")
    analyze_auroc(
        json_path=f"{config.output_dir}/inference_results.json",
        output_path=f"{config.output_dir}/uncertainty_auroc.png",
        logger=logger
    )
    logger.info(f"Saved AUROC plot to {config.output_dir}/uncertainty_auroc.png")

if __name__ == "__main__":
    main()