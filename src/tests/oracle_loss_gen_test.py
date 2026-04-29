from transformers import AutoTokenizer, AutoModelForCausalLM
from omegaconf import DictConfig, OmegaConf
import h5py
import os
import torch
import logging

from typing import List, Dict, Tuple, Union, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from tqdm import tqdm
import json

from src.prompt import GENERATE_PROMPT
from src.utils import (
    setup_logger, load_config, load_qa_dataset, load_h5_scores,
    apply_template, compute_metrics,
    CtxExample, QAExample,
    InferenceResult,
    validate_and_save_results
)

logger = logging.getLogger(__name__)

def rerank_contexts(
    oracle_scores,
    idx: str,
    ctxs: List[CtxExample],
):
    # Prepare pairs for reranking
    ctxs = ctxs[10:]    # Skip the first 10 passages, only consider the retrieved ones
    # passages = [f"Title: {ctx.title}\n\n{ctx.text}" for ctx in ctxs]
    sample_oracle_scores = oracle_scores[idx][10:]  # Skip the first 10 passages
    _, indices = torch.sort(sample_oracle_scores, descending=True)
    reranked_ctxs = [ctxs[idx] for idx in indices]
    return reranked_ctxs

def construct_context(
    ctxs: List[CtxExample],
    use_single_context: bool = True,
    topk: int = -1,
    do_rerank: bool = False,
    idx: str = None,
    oracle_scores = None
) -> str:
    if not ctxs:
        return ""
    elif use_single_context:
        target_ctx = ctxs[0]
        context = f"Title: {target_ctx.title}\n\n{target_ctx.text}"
        return context
    else:
        contexts = []
        if do_rerank:
            assert oracle_scores is not None, "Oracle scores must be provided for reranking contexts."
            ctxs = rerank_contexts(oracle_scores, idx, ctxs)
        else:
            ctxs = ctxs[10:]    # Skip the first 10 passages, only consider the retrieved ones
        ctxs = ctxs[:topk] if topk > 0 else ctxs

        for ctx in ctxs:
            context = f"Title: {ctx.title}\n\n{ctx.text}"
            contexts.append(context)
        return "\n\n".join(contexts)
    

def run_test_oracle_inference(
    config: DictConfig,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    data: List[QAExample],
    oracle_scores = None
) -> List[InferenceResult]:
    results = []
    generate_prompt = GENERATE_PROMPT[config.generate_prompt_name]

    for idx, item in tqdm(enumerate(data), total=len(data), desc="Running Oracle Loss Generation Test"):
        query = item.question
        # Compare
        # if item.pseudo_answer is None:
        #     continue

        context = construct_context(
            item.ctxs,
            config.data.use_single_context,
            topk=config.data.topk_per_query,
            do_rerank=config.data.do_rerank,
            idx = item.idx,
            oracle_scores=oracle_scores
        )
        query_text = generate_prompt.format(question=item.question)
        input_text = apply_template(query_text, context, config.model.model_name)

        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(model.device)
        attention_mask = torch.ones_like(input_ids).to(model.device)
        outputs = model.generate(input_ids, attention_mask=attention_mask, pad_token_id=tokenizer.pad_token_id, **config.model.gen_kwargs)

        # Decode generated answer
        gen_ids = outputs[:, input_ids.shape[1]:-1]
        pred_answer = tokenizer.decode(gen_ids[0])

        answers = item.answers
        if isinstance(answers, dict):
            answers = answers.get("aliases", None)  # TriviaQA format
        metrics = compute_metrics(pred_answer, answers)

        result = InferenceResult(
            id=idx,
            question=query,
            pred_answer=pred_answer,
            answers=answers,
            metrics=metrics
        )
        results.append(result)
    return results


def main():
    config = load_config()
    cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    experiment_name = f"prompt={config.generate_prompt_name}"
    output_dir = os.path.join(config.output_dir, config.data.name)  # Use data name from config
    config.output_dir = os.path.join(output_dir, experiment_name, cur_time)
    
    setup_logger(f"oracle_loss_test_inference_{cur_time}", config.output_dir)
    logger = logging.getLogger(__name__)
    logger.info("Configuration Loaded:")
    logger.info(OmegaConf.to_yaml(config))

    # Load data
    data = load_qa_dataset(config.data.data_path)
    logger.info(f"Loaded {len(data)} data entries from {config.data.data_path}")

    # Initialize model
    model = AutoModelForCausalLM.from_pretrained(config.model.model_name, torch_dtype="bfloat16", attn_implementation="flash_attention_2")
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Model {config.model.model_name} initialized.")

    oracle_scores = load_h5_scores(config.data.precompute_table_path)
    inference_results = run_test_oracle_inference(config, model, tokenizer, data, oracle_scores)
    validate_and_save_results(inference_results, config.output_dir, logger)


if __name__ == "__main__":
    main()