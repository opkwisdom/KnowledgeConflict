from typing import Any, Dict, List, Tuple, Union, Optional
import torch
from omegaconf import OmegaConf, DictConfig
from datetime import datetime
from dataclasses import dataclass, asdict
from tqdm import tqdm
import logging
import json
import os

from KVzip.model import ModelKVzip
from kfc_model import KnowledgeFusionCore
from judge_model import LLMJudger, OpenAIJudger, HfLLMJudger, JudgeOutput
from prompt import ALL_PROMPTS, PSEUDO_PASSAGE_PROMPT, GENERATE_PROMPT
from utils import (
    setup_logger, load_config, load_relevance_dataset, compute_metrics,
    MetricResult, RelevanceQAExample
)


@dataclass
class InferenceResult:
    id: int
    question: str
    pred_answer: str
    answers: List[str]
    metrics: MetricResult


def run_inference(
    config: DictConfig,
    kfc: KnowledgeFusionCore,
    llm_judger: LLMJudger,
    data: List[RelevanceQAExample],
    logger,
) -> Dict[str, List[InferenceResult]]:
    inference_cases = ["param_true", "param_positive", "param_negative", "param_irrelevant", "param_multiple"]
    results = {infer_case: [] for infer_case in inference_cases}

    for idx, item in tqdm(enumerate(data), desc="Running KFC Inference", total=len(data)):
        question = item.question
        a_internal = kfc.generate_internal_answer(question)

        if idx == 0:
            logger.info(f"Sample Internal Answer: {a_internal}")
        
        judge_output: JudgeOutput = llm_judger.judge(
            query=item.question,
            answer=a_internal,
            contexts=item.ctxs,
        )
        item.ctx_relevance = judge_output.ctx_relevance     # Real-time LLM judged relevance
        # is_correct = llm_check_answer(query, a_internal, item.ctxs)
        # is_correct = check_answer(a_internal, item.answers)     # Naive check

        is_correct = judge_output.is_correct
        # Case 1 - Internal answer is correct
        if is_correct:
            sample_result = InferenceResult(
                id=idx,
                question=item.question,
                pred_answer=a_internal,
                answers=item.answers,
                metrics=compute_metrics(a_internal, item.answers),
            )
            results["param_true"].append(sample_result)
        # Case 2 - Internal answer is incorrect
        else:
            # Facade pattern
            pred_answer, rel_type = kfc.resolve_and_generate(
                query=item.question,
                contexts=item.ctxs,
                relevance=item.ctx_relevance,
                internal_answer=a_internal,
                use_single_context=True,    # Temporary
            )

            sample_result = InferenceResult(
                id=idx,
                question=item.question,
                pred_answer=pred_answer,
                answers=item.answers,
                metrics=compute_metrics(pred_answer, item.answers),
            )
            results[f"param_{rel_type}"].append(sample_result)
    logger.info("Inference completed.")
    # logger.info(f"Total cost: {llm_judger.get_total_cost():.6f} USD")

    return results

def validate_and_save_results(
    results: Dict[str, List[InferenceResult]],
    output_dir: str,
    logger: logging.Logger,
) -> None:
    summary_path = f"{output_dir}/inference_summary.txt"
    all_results_path = f"{output_dir}/inference_results.json"
    summary = {}

    for k, inference_list in results.items():
        total = len(inference_list)

        correct = sum([1 for res in inference_list if res.metrics.soft_em])
        recall = sum([res.metrics.recall for res in inference_list]) / total if total > 0 else 0.0
        precision = sum([res.metrics.precision for res in inference_list]) / total if total > 0 else 0.0
        f1 = sum([res.metrics.f1 for res in inference_list]) / total if total > 0 else 0.0

        accuracy = correct / total if total > 0 else 0.0
        logger.info(f"Case {k}: Total={total}, Correct={correct}, Accuracy={accuracy:.4f},"
                    f" Recall={recall:.4f}, Precision={precision:.4f}, F1={f1:.4f}")
        summary[k] = {
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
        json_results = {
            k: [asdict(res) for res in v] for k, v in results.items()
        }
        json.dump(json_results, f, ensure_ascii=False, indent=4)

def main():
    config = load_config()
    cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    expriment_name = f"ratio={config.model.prune.ratio}_prompt={config.generate_prompt_name}"
    config.output_dir = os.path.join(config.output_dir, expriment_name, cur_time)
    logger = setup_logger(f"main_{cur_time}", config.output_dir)
    logger.info("Configuration Loaded:")
    logger.info(OmegaConf.to_yaml(config))

    repeat_prompt = None
    generate_prompt = None

    if config.task != "pseudo-passage":
        repeat_prompt = ALL_PROMPTS[config.self_task_prompt_name]
    else:
        pair_prompt = PSEUDO_PASSAGE_PROMPT[config.self_task_prompt_name]
        # generate_prompt = pair_prompt["generate"]
        repeat_prompt = pair_prompt["repeat"]
    
    base_prompt = GENERATE_PROMPT["pure-llm-brief-2"]
    generate_prompt = GENERATE_PROMPT[config.generate_prompt_name]

    logger.info(f"Using base prompt: {base_prompt}")

    # Load data
    data = load_relevance_dataset(config.data.data_path)
    logger.info(f"Loaded {len(data)} data entries from {config.data.data_path}")

    # Initialize model
    kvzip = ModelKVzip(config.model.model_name, gen_kwargs=config.model.gen_kwargs, prompt=repeat_prompt)
    logger.info(f"Model {config.model.model_name} initialized.")
    kfc = KnowledgeFusionCore(config, kvzip, generate_prompt, base_prompt, logger)
    logger.info("Knowledge Fusion Core initialized.")

    if config.judger.use_openai:
        judger: LLMJudger = OpenAIJudger(config.judger)
    else:
        judger: LLMJudger = HfLLMJudger(config.judger)

    inference_result = run_inference(config, kfc, judger, data, logger)
    validate_and_save_results(inference_result, config.output_dir, logger)

if __name__ == "__main__":
    main()