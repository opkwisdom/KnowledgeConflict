import os
import json
import yaml
import torch
from typing import Union, List, Dict, Tuple
from omegaconf import OmegaConf, DictConfig
from argparse import ArgumentParser
from transformers import LlamaModel
from datetime import datetime
from dataclasses import dataclass, asdict
import pickle
from tqdm import tqdm

from utils import load_config, load_json_data, setup_logger, has_answer
from prompt import ALL_PROMPTS, PSEUDO_PASSAGE_PROMPT, GENERATE_PROMPT
from KVzip.model import ModelKVzip
from KVzip.attention.kvcache import RetainCache, EvictCache

@dataclass
class TestEx:
    id: int
    question: str
    a_internal: str
    answers: List[str]
    ctx_idx: int
    ctx_rel: str
    kv_cache_score: torch.Tensor


def grade_answer(a_pred: str, a_true: List[str]) -> bool:
    """Simple grading function to compare predicted and true answers."""
    return any(ans.strip().lower() in a_pred.strip().lower() for ans in a_true)


def detect_conflict_base(
    config: DictConfig,
    model: ModelKVzip,
    item: dict,
    ctx_idx: int,
    prompt: str,
    topk: int,
    logger
) -> Tuple[torch.Tensor, str, bool]:
    # Generate internal answer (or use extracted answer)
    if config.do_answer:
        user_content = prompt.format(question=item['question'])
        messages = [
            {"role": "user", "content": user_content}
        ]
        q_ids = model.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_special_tokens=True,
        ).to(model.device)
        inputs = {
            "input_ids": q_ids,
            "attention_mask": torch.ones_like(q_ids).to(model.device),
        }
        with torch.no_grad():
            raw_ids = model.model.generate(**inputs, **model.gen_kwargs)
        generated_ids = raw_ids[:, q_ids.shape[1]:]
        a_internal = model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    else:
        a_internal = item["parametric_answer"]

    is_correct = has_answer(a_internal, item['answers'])

    # These two inputs are used for detecting conflict
    q_ids = model.encode(f"Question: {item['question']}\n")
    a_ids = model.encode(f"Answer: {a_internal}\n")
    
    # Detect conflict by repeated generation
    target_ctx = item['ctxs'][ctx_idx]
    context = f"Title: {target_ctx['title']}\n\n{target_ctx['text']}"
    kv = model.prefill(
        context,
        q_ids,
        a_ids,
        load_score=False,
        do_score=True,
    )
    kv.to("cpu")

    # Score analysis
    n_layer = len(kv.raw_score)
    sample_topk_diff_scores = torch.zeros((n_layer, topk), dtype=torch.float32)

    for i, layer_score in enumerate(kv.raw_score):
        score_flat = layer_score.reshape(-1)
        topk_scores = torch.topk(score_flat, k=topk, largest=True).values
        mean_score = torch.mean(score_flat)
        topk_diff_scores = topk_scores - mean_score
        sample_topk_diff_scores[i] = topk_diff_scores
    
    return sample_topk_diff_scores, a_internal, is_correct


def detect_conflict_pseudo_passage(
    config: DictConfig,
    model: ModelKVzip,
    item: dict,
    ctx_idx: int,
    prompt: str,
    generate_prompt: str,
    topk: int,
    logger
) -> Tuple[torch.Tensor, str, bool]:
    # Generate pseudo-passage
    prompt = generate_prompt.format(question=item['question'])
    g_ids = model.apply_template(prompt)
    pseudo_passage_ids = model.model.generate(g_ids, **model.gen_kwargs)[:, g_ids.shape[1]:]
    pseudo_passage = model.tokenizer.decode(pseudo_passage_ids[0], skip_special_tokens=True)
    is_correct = grade_answer(pseudo_passage, item['answers'])

    # Detect conflict by repeated generation
    target_ctx = item['ctxs'][ctx_idx]
    context = f"Title: {target_ctx['title']}\n\n{target_ctx['text']}"
    ctx_ids = model.encode(context)

    kv = model.prefill(
        ctx_ids=pseudo_passage_ids,
        q_ids=ctx_ids,
        a_ids=None,
        load_score=False,
        do_score=True,
    )
    kv.to("cpu")

    # Score analysis
    n_layer = len(kv.raw_score)
    sample_topk_diff_scores = torch.zeros((n_layer, topk), dtype=torch.float32)

    for i, layer_score in enumerate(kv.raw_score):
        score_flat = layer_score.reshape(-1)
        topk_scores = torch.topk(score_flat, k=topk, largest=True).values
        mean_score = torch.mean(score_flat)
        topk_diff_scores = topk_scores - mean_score
        sample_topk_diff_scores[i] = topk_diff_scores

    return sample_topk_diff_scores, pseudo_passage, is_correct


def test_hypothesis(config: DictConfig, model: ModelKVzip, data: list, output_dir: str, generate_prompt: str = None, logger = None):
    case_num = 0

    test_results = {
        "true_rel": [],
        "true_neg": [],
        "true_irr": [],
        "false_rel": [],
        "false_neg": [],
        "false_irr": [],
    }

    prompt = GENERATE_PROMPT[config.generate_prompt_name]
    logger.info(f"Using prompt:\n{prompt}")
    
    # Select detection function
    task_map = {
        "pseudo-passage": detect_conflict_pseudo_passage,
        "base": detect_conflict_base,
    }
    detect_conflict_func = task_map.get(config.task, detect_conflict_base)
    is_pseudo_passage = (detect_conflict_func == detect_conflict_pseudo_passage)

    # Metadata for loop
    context_types = [
        ("positive", "relevant", "rel"),
        ("negative", "negative", "neg"),
        ("irrelevant", "irrelevant", "irr"),
    ]

    for i, item in tqdm(enumerate(data), total=len(data), desc="Testing hypothesis"):
        rel = item["ctx_relevance"]
        if not all(len(rel[c_key]) > 0 for c_key, _, _ in context_types):
            continue

        case_num += 1

        # Prepare KV cache with only one context
        for ctx_key, label, result_suffix in context_types:
            for ctx_idx in rel[ctx_key]:
                # Send to detection function as positional arguments
                func_args = [config, model, item, ctx_idx, prompt]
                if is_pseudo_passage:
                    func_args.append(generate_prompt)
                func_args.append([config.topk, logger])

                kv_score, a_internal, is_correct = detect_conflict_func(*func_args)
                test_ex = TestEx(
                    id=i,
                    question=item['question'],
                    a_internal=a_internal,
                    answers=item['answers'],
                    ctx_idx=ctx_idx,
                    ctx_rel=label,
                    kv_cache_score=kv_score,
                )
                status = "true" if is_correct else "false"
                test_results[f"{status}_{result_suffix}"].append(test_ex)
    
    logger.info(f"Total hypothesis test cases: {case_num}")
    for key, lst in test_results.items():
        logger.info(f"{key}: {len(lst)} examples")
    
    output_path = os.path.join(output_dir, f"kv_cache_hypothesis_test_results.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(test_results, f)
    logger.info(f"Saved test results to {output_path}")


def main():
    # Load configuration
    config = load_config()
    cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    repeat_prompt = None
    generate_prompt = None

    if config.task != "pseudo-passage":
        repeat_prompt = ALL_PROMPTS[config.prompt_name]
    else:
        pair_prompt = PSEUDO_PASSAGE_PROMPT[config.prompt_name]
        generate_prompt = pair_prompt["generate"]
        repeat_prompt = pair_prompt["repeat"]
    
    config.output_dir = os.path.join(config.output_dir, f"{config.prompt_name}_raw", cur_time)     # for different prompts
    logger = setup_logger(f"kv_cache_hypothesis_test_{cur_time}", config.output_dir)
    os.makedirs(config.output_dir, exist_ok=True)
    
    logger.info("Configuration Loaded:")
    logger.info(OmegaConf.to_yaml(config))

    # Load data
    data = load_json_data(config.data.data_path)
    logger.info(f"Loaded {len(data)} data entries from {config.data.data_path}")

    # Initialize model
    model = ModelKVzip(config.model.model_name, gen_kwargs=config.model.gen_kwargs, prompt=repeat_prompt, logger=logger)
    logger.info(f"Model {config.model.model_name} initialized.")

    # generate_pseudo_passage(config, model, data, config.output_dir, generate_prompt, logger)
    test_hypothesis(config, model, data, config.output_dir, generate_prompt, logger)

if __name__ == "__main__":
    main()