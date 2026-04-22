from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils import (
    load_json_data, QAExample
)
import json
import torch
import sys
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

def prepare_ctxs(supporting_facts, ctxs):
    gold_ctxs, neg_ctxs = [], []
    titles = ctxs["title"]
    sentences_list = ctxs["sentences"]
    gold_titles = supporting_facts["title"]
    
    for title, sentences in zip(titles, sentences_list):
        sentence_text = " ".join(sentences)
        ctx = f"Title: {title}\n\n{sentence_text}"
        
        if title in gold_titles:
            gold_ctxs.append(ctx)
        else:
            neg_ctxs.append(ctx)
            
    return gold_ctxs, neg_ctxs
### Helper functions ###

def prepare_inputs(data, tokenizer, test_type):
    all_full_texts = []
    metadata = []   # (item_idx, ctx_type, prompt_len)
    valid_items = {}

    for idx, item in enumerate(tqdm(data, desc="Flattening Data")):
        supporting_facts = item["supporting_facts"]
        ctxs = item["context"]
        gold_ctxs, neg_ctxs = prepare_ctxs(supporting_facts, ctxs)
        if not gold_ctxs:
            continue

        question = item["question"]
        answers = item["answer"] if test_type == "base" else item["pseudo_answer"]
        if answers is None:
            continue
        if isinstance(answers, list):
            answers = ", ".join(answers)

        valid_items[idx] = {"question": question, "answers": answers}
        ft, p_len = prepare_single_text(None, question, answers, tokenizer)
        all_full_texts.append(ft)
        metadata.append((idx, "base", p_len))

        for ctx in gold_ctxs:
            ft, p_len = prepare_single_text(ctx, question, answers, tokenizer)
            all_full_texts.append(ft)
            metadata.append((idx, "gold", p_len))
        
        for ctx in neg_ctxs:
            ft, p_len = prepare_single_text(ctx, question, answers, tokenizer)
            all_full_texts.append(ft)
            metadata.append((idx, "neg", p_len))

    print(f"Total prompts to process: {len(all_full_texts)}")
    return all_full_texts, metadata, valid_items

def reconstruct_results(outputs, metadata, valid_items):
    results_dict = defaultdict(lambda: {"base_loss": [], "gold_loss": [], "neg_loss": []})

    for out, meta in zip(outputs, metadata):
        idx, ctx_type, prompt_len = meta
        loss = compute_nll_loss(out, prompt_len)
        results_dict[idx][f"{ctx_type}_loss"].append(loss)

    results = []
    for idx, res in results_dict.items():
        results.append({
            "question": valid_items[idx]["question"],
            "answers": valid_items[idx]["answers"],
            "base_loss": res["base_loss"],
            "gold_loss": res["gold_loss"],
            "neg_loss": res["neg_loss"],
        })
    return results


def main():
    test_type = sys.argv[1]
    data = load_json_data("data/hotpotqa-w/raw/validation_p.jsonl")
    data = data[:1000]   # Limit to first 1000 examples for testing
    model_name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", torch_dtype="auto")
    num_device = torch.cuda.device_count()
    assert num_device % 2 == 0, "Number of GPUs must be even for tensor parallelism."

    model = LLM(model_name_or_path, tensor_parallel_size=num_device, disable_log_stats=True, gpu_memory_utilization=0.7)
    sampling_params = SamplingParams(
        temperature=0.0,
        prompt_logprobs=1,
        max_tokens=1,
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    )

    print("Preparing inputs for generation loss calculation...")
    all_full_texts, metadata, valid_items = prepare_inputs(data, tokenizer, test_type)
    print("Calculating generation losses...")
    batch_size = 128
    outputs = []
    for i in tqdm(range(0, len(all_full_texts), batch_size), desc="Processing batches"):
        batch_texts = all_full_texts[i : i + batch_size]
        batch_outputs = model.generate(batch_texts, sampling_params, use_tqdm=False)
        outputs.extend(batch_outputs)
    print("Reconstructing results...")
    results = reconstruct_results(outputs, metadata, valid_items)
    print(f"Calculated generation losses for {len(results)} examples.")
    
    with open(f"src/tests/oracle_loss/{test_type}_vllm_genloss_test_result.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()