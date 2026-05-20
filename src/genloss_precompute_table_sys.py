from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import (
    load_json_data, apply_template, QAExample,
    setup_logger
)
from prompt import GENERATE_PROMPT

import json
import torch
import sys
import os
import h5py
import logging
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
from typing import List, Dict, Tuple, Union
from dataclasses import asdict
from collections import defaultdict
from vllm import LLM, SamplingParams


BASE_TEMPLATE = GENERATE_PROMPT["pure-llm"]
CONTEXT_TEMPLATE = GENERATE_PROMPT["base"]
# MODEL_NAME_OR_PATH = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_NAME_OR_PATH = "Qwen/Qwen2.5-7B-Instruct"


### Helper functions ###
def prepare_single_text(ctx_str: Union[None, List[str]], question, answers, tokenizer):
    if ctx_str is None:
        query_text = BASE_TEMPLATE.format(question=question)
        input_text = apply_template(query_text, None, MODEL_NAME_OR_PATH)
    else:
        ctx_tokens = tokenizer.encode(ctx_str, add_special_tokens=False)
        if len(ctx_tokens) > 512:
            ctx_tokens = ctx_tokens[:512]
            ctx_str = tokenizer.decode(ctx_tokens)
        query_text = CONTEXT_TEMPLATE.format(question=question)
        input_text = apply_template(query_text, ctx_str, MODEL_NAME_OR_PATH)
    
    full_text = input_text + answers + tokenizer.eos_token
    input_ids = tokenizer(input_text, add_special_tokens=False).input_ids
    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
    

    if full_ids[:len(input_ids)] != input_ids:
        prompt_len = len(input_ids)
    else:
        prompt_len = len(input_ids)
    
    return full_text, prompt_len

def compute_nll_loss(output, prompt_len):
    prompt_logprobs = output.prompt_logprobs
    token_ids = output.prompt_token_ids
    target_logprobs = []
    
    for i in range(prompt_len, len(prompt_logprobs)):
        token_dict = prompt_logprobs[i]
        if token_dict is None:
            continue
        actual_token_id = token_ids[i]
        logprob = token_dict[actual_token_id].logprob
        target_logprobs.append(logprob)
        
    if target_logprobs:
        return -sum(target_logprobs) / len(target_logprobs)
    return float('inf')

def prepare_ctxs(ctxs) -> List[str]:
    total_ctxs = []
    for ctx in ctxs:
        ctx_str = f"Title: {ctx['title']}\n\n{ctx['text']}"
        total_ctxs.append(ctx_str)
            
    return total_ctxs
### Helper functions ###

def prepare_inputs(data, tokenizer):
    all_full_texts = []
    metadata = []   # (item_idx, ctx_type, prompt_len), only consider base, doc
    valid_items = {}

    for _, item in enumerate(tqdm(data, desc="Flattening Data")):
        idx = item["idx"]
        ctxs = item["ctxs"]
        # In contrast to the test set, we don't need to distinguish between gold and negative contexts
        total_ctxs = prepare_ctxs(ctxs)

        question = item["question"]
        answers = item["answers"]
        # answers = item["answer"] if test_type == "base" else item["pseudo_answer"]
        if answers is None:
            continue
        if isinstance(answers, list):
            answers = ", ".join(answers)

        valid_items[idx] = {"question": question, "answers": answers}
        ft, p_len = prepare_single_text(None, question, answers, tokenizer)
        all_full_texts.append(ft)
        metadata.append((idx, "base", p_len))

        for ctx in total_ctxs:
            ft, p_len = prepare_single_text(ctx, question, answers, tokenizer)
            all_full_texts.append(ft)
            metadata.append((idx, "doc", p_len))

    print(f"Total prompts to process: {len(all_full_texts)}")
    return all_full_texts, metadata, valid_items

def reconstruct_results(outputs, metadata, valid_items):
    results_dict = defaultdict(lambda: {"base_loss": [], "doc_loss": []})

    for out, meta in zip(outputs, metadata):
        idx, ctx_type, prompt_len = meta
        loss = compute_nll_loss(out, prompt_len)
        results_dict[idx][f"{ctx_type}_loss"].append(loss)

    results = []
    for idx, res in results_dict.items():
        results.append({
            "idx": idx,
            "base_loss": res["base_loss"],
            "doc_loss": res["doc_loss"]
        })
    return results

def safe_open_h5(local_rank, filepath, mode="a"):
    try:
        return h5py.File(filepath, mode)
    except OSError as e:
        print(f"[Rank {local_rank}] 파일 손상 감지됨: {filepath}. 삭제 후 재생성합니다.")
        if os.path.exists(filepath):
            os.remove(filepath)
        return h5py.File(filepath, mode)

def main():
    parser = ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    # input_dir = "/workspaces/kvzip_nlplab/data/hotpotqa-w/retrieved"
    input_dir = "/workspaces/kvzip_nlplab/data/train"
    # output_dir = "/workspaces/kvzip_nlplab/checkpoint/genloss_precompute_table/llama"
    output_dir = "/workspaces/kvzip_nlplab/checkpoint/genloss_precompute_table/qwen"
    input_path = os.path.join(input_dir, args.input_file)
    save_path = os.path.join(output_dir, args.output_file)

    setup_logger("main", output_dir)
    logger = logging.getLogger(__name__)

    
    data = load_json_data(input_path)
    # model_name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    model_name_or_path = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    num_device = torch.cuda.device_count()
    assert num_device > 0, "No GPUs detected. Please run on a machine with at least one GPU."

    model = LLM(
        model_name_or_path,
        tensor_parallel_size=num_device,
        disable_log_stats=True,
        gpu_memory_utilization=0.8,
        max_model_len=1024,
        enforce_eager=True
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        prompt_logprobs=1,
        max_tokens=1,
        detokenize=False,
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    )

    logger.info("Preparing inputs for generation loss calculation...")
    all_full_texts, metadata, valid_items = prepare_inputs(data, tokenizer)
    logger.info(f"Calculating generation losses for {len(valid_items)}...")
    
    batch_size = 1024
    SAVE_QUERY_SIZE = 100
    os.makedirs(save_path, exist_ok=True)

    valid_indices = list(valid_items.keys())
    total_queries = len(valid_indices)
    logger.info(f"Total valid examples to process: {total_queries}")

    for chunk_start in tqdm(range(0, total_queries, SAVE_QUERY_SIZE), desc="Processing chunks"):
        chunk_end = min(chunk_start + SAVE_QUERY_SIZE, total_queries)
        
        chunk_indices = set(valid_indices[chunk_start:chunk_start + SAVE_QUERY_SIZE])
        chunk_save_path = os.path.join(save_path, f"chunk_{chunk_start}_to_{chunk_end}.h5")
        if os.path.exists(chunk_save_path):
            logger.info(f"Chunk {chunk_start} to {chunk_end} already exists. Skipping...")
            continue
        
        chunk_texts = []
        chunk_metadata = []
        for text, meta in zip(all_full_texts, metadata):
            if meta[0] in chunk_indices:
                chunk_texts.append(text)
                chunk_metadata.append(meta)
        logger.info(f"Processing chunk {chunk_start} to {chunk_end} with {len(chunk_texts)} prompts...")
        chunk_outputs = []

        for i in tqdm(range(0, len(chunk_texts), batch_size), desc="Processing batches in chunk", leave=False):
            batch_texts = chunk_texts[i : i + batch_size]
            batch_outputs = model.generate(batch_texts, sampling_params, use_tqdm=False)
            chunk_outputs.extend(batch_outputs)
        chunk_results = reconstruct_results(chunk_outputs, chunk_metadata, valid_items)

        with h5py.File(chunk_save_path, 'w') as h5file:
            for item in chunk_results:
                grp_name = str(item['idx'])
                # if grp_name in h5file:
                #     del h5file[grp_name]
                grp = h5file.create_group(grp_name)
                grp.create_dataset("base_loss", data=np.array(item["base_loss"], dtype=np.float32))
                grp.create_dataset("doc_loss", data=np.array(item["doc_loss"], dtype=np.float32))
        logger.info(f"Saved chunk {chunk_start} to {chunk_end} results to HDF5.")


if __name__ == "__main__":
    main()