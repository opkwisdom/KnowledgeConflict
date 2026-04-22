import h5py
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict
from omegaconf import OmegaConf, DictConfig
from dataclasses import asdict
import os
import json
import torch
import numpy as np
import logging

from src.utils import (
    load_config, setup_logger, load_qa_dataset,
    apply_template, compute_metrics,
    QAExample, InferenceResult, CtxExample
)
from src.prompt import GENERATE_PROMPT


def rerank_contexts(
    query_id: str,
    ctxs: List[CtxExample],
    precompute_table: dict,
    score_mode: str = "marginal-qa",
    min_k: int = 3,
    threshold: float = 0.0
) -> List[CtxExample]:
    # query_id = str(idx)
    if query_id not in precompute_table:
        return ctxs[:min_k]

    scores = precompute_table[query_id][score_mode][:]
    rerank_indices = np.argsort(scores)[::-1]
    supportive_indices = [i for i in rerank_indices if scores[i] > threshold]
    if not supportive_indices:
        return ctxs[:min_k]

    return [ctxs[i] for i in supportive_indices]

def construct_context(
    idx: str,
    ctxs: List[CtxExample],
    use_single_context: bool = False,
    topk: int = -1,
    do_rerank: bool = True,
    precompute_table: dict = None,
    score_mode: str = "oracle"
) -> str:
    if use_single_context:
        target_ctx = ctxs[0]
        context = f"Title: {target_ctx.title}\n\n{target_ctx.text}"
        return context
    else:
        contexts = []
        if do_rerank:
            ctxs = rerank_contexts(idx, ctxs, precompute_table, score_mode)
        ctxs = ctxs[:topk] if topk > 0 else ctxs

        for ctx in ctxs:
            contexts.append(f"Title: {ctx.title}\n\n{ctx.text}")
        return "\n\n".join(contexts)

def run_inference(
    config: DictConfig,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    precompute_table: dict,
    data: List[QAExample]
) -> List[InferenceResult]:
    logger = logging.getLogger(__name__)
    logger.info("Starting inference on the dataset...")
    results = []
    generate_prompt = GENERATE_PROMPT[config.generate_prompt_name]

    for idx, item in tqdm(enumerate(data), desc="Inference Progress", total=len(data)):
        query_id = item.idx
        context = construct_context(
            query_id,
            item.ctxs,
            config.data.use_single_context,
            topk=config.data.topk_per_query,
            do_rerank=config.data.do_rerank,
            precompute_table=precompute_table,
            score_mode=config.data.score_mode
        )
        query_text = generate_prompt.format(question=item.question)
        input_text = apply_template(query_text, context, config.model.model_name)

        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(model.device)
        attention_mask = torch.ones_like(input_ids).to(model.device)
        outputs = model.generate(input_ids, attention_mask=attention_mask, pad_token_id=tokenizer.pad_token_id, **config.model.gen_kwargs)

        # Decode generated answer
        gen_ids = outputs[:, input_ids.shape[1]:]
        pred_answer = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

        answers = item.answers
        metrics = compute_metrics(pred_answer, answers)

        # Construct result
        sample_result = InferenceResult(
            id=idx,
            question=item.question,
            pred_answer=pred_answer,
            answers=answers,
            metrics=metrics,
        )
        results.append(sample_result)
    
    return results


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


def main():
    # Load the model and tokenizer
    config = load_config()
    cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = os.path.join(config.output_dir, config.data.name)  # Use data name from config
    config.output_dir = os.path.join(output_dir, config.experiment_name, cur_time)
    
    setup_logger(f"oracle_inference_{cur_time}", config.output_dir)
    logger = logging.getLogger(__name__)
    logger.info("Configuration Loaded:")
    logger.info(OmegaConf.to_yaml(config))

    # Load data
    precompute_table = {}
    with h5py.File(config.data.precompute_table_path, 'r') as f:
        for sample_id in f.keys():
            precompute_table[sample_id] = {}
            for score_name in f[sample_id].keys():
                precompute_table[sample_id][score_name] = f[sample_id][score_name][:]
        
    data = load_qa_dataset(config.data.data_path)
    logger.info(f"Loaded {len(data)} data entries from {config.data.data_path}")

    # Initialize model
    model = AutoModelForCausalLM.from_pretrained(config.model.model_name, torch_dtype="bfloat16", attn_implementation="flash_attention_2")
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
    tokenizer.pad_token_id = 128004
    tokenizer.padding_side = "left"
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    # model = None
    # tokenizer = None
    logger.info(f"Model {config.model.model_name} initialized.")

    # Inference
    inference_results = run_inference(config, model, tokenizer, precompute_table, data)
    validate_and_save_results(inference_results, config.output_dir, logger)


if __name__ == "__main__":
    main()