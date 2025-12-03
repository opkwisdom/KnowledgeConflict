from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import DictConfig, OmegaConf
from typing import List, Dict, Tuple, Union, Optional
import torch
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from tqdm import tqdm
import json
import os

from src.prompt import GENERATE_PROMPT
from src.utils import (
    load_config, setup_logger, load_relevance_dataset, check_answer,
    apply_template,
    RelevanceQAExample, CtxExample,
    InferenceResult,
)

def construct_context(
    ctxs: List[CtxExample],
    relevance_map: Dict[int, str],
    use_single_context: bool = True,
) -> Tuple[str, str]:
    if not ctxs:
        return ""
    elif use_single_context:
        target_ctx = ctxs[0]
        context = f"Title: {target_ctx.title}\n\n{target_ctx.text}"
        return context, relevance_map[0]
    else:
        contexts = []
        for ctx in ctxs:
            contexts.append(f"Title: {ctx.title}\n\n{ctx.text}")
        return "\n\n".join(contexts), "multiple"


def run_inference(
    config: DictConfig,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    data: List[RelevanceQAExample],
    logger,
) -> Dict[str, List[InferenceResult]]:
    inference_cases = ["param_true", "param_positive", "param_negative", "param_irrelevant", "param_multiple"]
    results = {infer_case: [] for infer_case in inference_cases}
    generate_prompt = GENERATE_PROMPT[config.generate_prompt_name]

    for idx, item in tqdm(enumerate(data), desc="Running RAG Inference", total=len(data)):
        a_internal = item.parametric_answer
        is_correct = check_answer(a_internal, item.answers)
        # Case 1 - Internal answer is correct
        if is_correct:
            sample_result = InferenceResult(
                id=idx,
                question=item.question,
                pred_answer=a_internal,
                answers=item.answers,
                is_correct=is_correct,
            )
            results["param_true"].append(sample_result)
            continue
        relevance_map = item.ctx_relevance.mapping

        context, rel_type = construct_context(item.ctxs, relevance_map, config.data.use_single_context)
        query_text = generate_prompt.format(question=item.question)
        input_text = apply_template(query_text, context, config.model.model_name)

        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(model.device)
        attention_mask = torch.ones_like(input_ids).to(model.device)
        outputs = model.generate(input_ids, attention_mask=attention_mask, pad_token_id=tokenizer.pad_token_id, **config.model.gen_kwargs)

        # Decode generated answer
        gen_ids = outputs[:, input_ids.shape[1]:-1]
        pred_answer = tokenizer.decode(gen_ids[0])

        is_correct = check_answer(pred_answer, item.answers)
        
        # Construct result
        sample_result = InferenceResult(
            id=idx,
            question=item.question,
            pred_answer=pred_answer,
            answers=item.answers,
            is_correct=is_correct,
        )
        results[f"param_{rel_type}"].append(sample_result)
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
        correct = sum([1 for res in inference_list if res.is_correct])
        accuracy = correct / total if total > 0 else 0.0
        logger.info(f"Case {k}: Total={total}, Correct={correct}, Accuracy={accuracy:.4f}")
        summary[k] = {
            "total": total,
            "correct": correct,
            "accuracy": round(accuracy, 4),
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

    experiment_name = f"prompt={config.generate_prompt_name}"
    config.output_dir = os.path.join(config.output_dir, experiment_name, cur_time)
    logger = setup_logger(f"rag_inference_{cur_time}", config.output_dir)
    logger.info("Configuration Loaded:")
    logger.info(OmegaConf.to_yaml(config))

    # Load data
    data = load_relevance_dataset(config.data.data_path)
    logger.info(f"Loaded {len(data)} data entries from {config.data.data_path}")

    # Initialize model
    model = AutoModelForCausalLM.from_pretrained(config.model.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Model {config.model.model_name} initialized.")

    # Inference
    # TODO: implement run_original_inference for QA performance comparison
    inference_results = run_inference(config, model, tokenizer, data, logger)
    validate_and_save_results(inference_results, config.output_dir, logger)

if __name__ == "__main__":
    main()