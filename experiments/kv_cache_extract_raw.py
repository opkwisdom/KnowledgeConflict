import os
import torch
from typing import Union, List, Dict, Tuple
from omegaconf import OmegaConf, DictConfig
from datetime import datetime
from dataclasses import dataclass, asdict
import pickle
from tqdm import tqdm
import logging

from utils import load_config, load_json_data, setup_logger, has_answer
from prompt import ALL_PROMPTS, PSEUDO_PASSAGE_PROMPT, GENERATE_PROMPT
from KVzip.model import ModelKVzip

@dataclass
class TestEx:
    id: int
    question: str
    a_internal: str
    answers: List[str]
    ctx_idx: int
    ctx_rel: str
    kv_cache_diff_score: torch.Tensor
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
    logger: logging.Logger
) -> Tuple[torch.Tensor, torch.Tensor, str, bool]:
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
    sampled_topk_scores = torch.zeros((n_layer, topk), dtype=torch.float32)

    for i, layer_score in enumerate(kv.raw_score):
        score_flat = layer_score.reshape(-1)
        topk_scores = torch.topk(score_flat, k=topk, largest=True).values
        mean_score = torch.mean(score_flat)
        topk_diff_scores = topk_scores - mean_score
        sample_topk_diff_scores[i] = topk_diff_scores
        sampled_topk_scores[i] = topk_scores
    
    return sample_topk_diff_scores, sampled_topk_scores, a_internal, is_correct


def test_hypothesis(
    config: DictConfig,
    model: ModelKVzip,
    data: list,
    output_dir: str,
    logger: logging.Logger = None,
    psg_num: int = 0
) -> None:
    case_num = 0

    test_results = {psg_num: []}

    prompt = GENERATE_PROMPT[config.generate_prompt_name]
    logger.info(f"Using prompt:\n{prompt}")

    for i, item in tqdm(enumerate(data), total=len(data), desc="Testing hypothesis"):
        func_args = [config, model, item, psg_num, prompt]
        func_args.extend([config.topk, logger])

        kv_diff_score, kv_score, a_internal, is_correct = detect_conflict_base(*func_args)
        test_ex = TestEx(
            id=i,
            question=item['question'],
            a_internal=a_internal,
            answers=item['answers'],
            ctx_idx=psg_num,
            ctx_rel=None,   # Does not apply here
            kv_cache_diff_score=kv_diff_score,
            kv_cache_score=kv_score,
        )
        test_results[psg_num].append(test_ex)
    
    logger.info(f"Total hypothesis test cases: {case_num}")
    for key, lst in test_results.items():
        logger.info(f"{key}: {len(lst)} examples")
    
    output_path = os.path.join(output_dir, f"kv_cache_results_{psg_num}.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(test_results, f)
    logger.info(f"Saved test results to {output_path}")


def main():
    # Load configuration
    config = load_config()
    cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    repeat_prompt = None
    generate_prompt = None

    repeat_prompt = ALL_PROMPTS[config.prompt_name]
    logger = setup_logger(f"kv_cache_extract_raw_{cur_time}", config.output_dir)
    os.makedirs(config.output_dir, exist_ok=True)
    
    logger.info("Configuration Loaded:")
    logger.info(OmegaConf.to_yaml(config))

    # Load data
    data = load_json_data(config.data.data_path)
    logger.info(f"Loaded {len(data)} data entries from {config.data.data_path}")

    # Initialize model
    model = ModelKVzip(config.model.model_name, gen_kwargs=config.model.gen_kwargs, prompt=repeat_prompt, logger=logger)
    logger.info(f"Model {config.model.model_name} initialized.")

    test_hypothesis(config, model, data, config.output_dir, logger=logger, psg_num=config.data.psg_num)

if __name__ == "__main__":
    main()