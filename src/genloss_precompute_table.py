from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils import (
    load_json_data, QAExample,
    setup_logger
)
import json
import torch
import sys
import os
import h5py
import logging
import numpy as np
from tqdm import tqdm
from dataclasses import asdict
from collections import defaultdict
from vllm import LLM, SamplingParams

QUESTION_TEMPLATE = "Question: {question}\n\n"
PROMPT_TEMPLATE = "Context:\n{ctx}\n\nQuestion: {question}\n\n"

### Helper functions ###
def prepare_single_text(ctx, question, answers, tokenizer):
    if ctx is None:
        prompt = QUESTION_TEMPLATE.format(question=question)
    else:
        prompt = PROMPT_TEMPLATE.format(ctx=ctx, question=question)
        
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    full_text = prompt_text + answers + tokenizer.eos_token
    prompt_len = len(tokenizer(prompt_text).input_ids)
    
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

def prepare_ctxs(ctxs):
    titles = ctxs["title"]
    sentences_list = ctxs["sentences"]
    total_ctxs = []
    
    for title, sentences in zip(titles, sentences_list):
        sentence_text = " ".join(sentences)
        ctx = f"Title: {title}\n\n{sentence_text}"
        total_ctxs.append(ctx)
            
    return total_ctxs
### Helper functions ###

def prepare_inputs(data, tokenizer):
    all_full_texts = []
    metadata = []   # (item_idx, ctx_type, prompt_len), only consider base, doc
    valid_items = {}

    for _, item in enumerate(tqdm(data, desc="Flattening Data")):
        idx = item["idx"]
        ctxs = item["context"]
        # In contrast to the test set, we don't need to distinguish between gold and negative contexts
        total_ctxs = prepare_ctxs(ctxs)

        question = item["question"]
        answers = item["pseudo_answer"]
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
    output_dir = "/workspaces/kvzip_nlplab/checkpoint/genloss_precompute_table"
    setup_logger("main", output_dir)
    logger = logging.getLogger(__name__)

    data = load_json_data("data/train/mhqa_train.jsonl")
    data = data[:1000]   # Limit to first 1000 examples for testing
    model_name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", torch_dtype="auto")
    num_device = torch.cuda.device_count()
    assert num_device > 0, "No GPUs detected. Please run on a machine with at least one GPU."
    assert num_device % 2 == 0, "Number of GPUs must be even for tensor parallelism."

    model = LLM(model_name_or_path, tensor_parallel_size=num_device, disable_log_stats=True, gpu_memory_utilization=0.7)
    sampling_params = SamplingParams(
        temperature=0.0,
        prompt_logprobs=1,
        max_tokens=1,
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    )

    logger.info("Preparing inputs for generation loss calculation...")
    all_full_texts, metadata, valid_items = prepare_inputs(data, tokenizer)
    logger.info(f"Calculating generation losses for {len(valid_items)}...")
    
    batch_size = 128
    SAVE_QUERY_SIZE = 10000
    save_path = "/workspaces/kvzip_nlplab/checkpoint/genloss_precompute_table/llama/loss_precompute_table.h5"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with h5py.File(save_path, 'w') as f:
        pass

    valid_indices = list(valid_items.keys())
    total_queries = len(valid_indices)
    logger.info(f"Total valid examples to process: {total_queries}")

    for chunk_start in tqdm(range(0, total_queries, SAVE_QUERY_SIZE), desc="Processing chunks"):
        chunk_indices = set(valid_indices[chunk_start:chunk_start + SAVE_QUERY_SIZE])
        chunk_texts = []
        chunk_metadata = []
        for text, meta in zip(all_full_texts, metadata):
            if meta[0] in chunk_indices:
                chunk_texts.append(text)
                chunk_metadata.append(meta)
        logger.info(f"Processing chunk {chunk_start} to {chunk_start + SAVE_QUERY_SIZE} with {len(chunk_texts)} prompts...")
        chunk_outputs = []

        for i in tqdm(range(0, len(chunk_texts), batch_size), desc="Processing batches in chunk", leave=False):
            batch_texts = chunk_texts[i : i + batch_size]
            batch_outputs = model.generate(batch_texts, sampling_params, use_tqdm=False)
            chunk_outputs.extend(batch_outputs)
        chunk_results = reconstruct_results(chunk_outputs, chunk_metadata, valid_items)

        with h5py.File(save_path, 'a') as h5file:
            for item in chunk_results:
                grp_name = str(item['idx'])
                if grp_name in h5file:
                    del h5file[grp_name]
                grp = h5file.create_group(grp_name)
                grp.create_dataset("base_loss", data=np.array(item["base_loss"], dtype=np.float32))
                grp.create_dataset("doc_loss", data=np.array(item["doc_loss"], dtype=np.float32))
        logger.info(f"Saved chunk {chunk_start} to {chunk_start + SAVE_QUERY_SIZE} results to HDF5.")


if __name__ == "__main__":
    main()