from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils import (
    load_json_data
)
from src.prompt import GENERATE_PROMPT
import json
import torch
import os
import sys
from tqdm import tqdm
from collections import defaultdict
from vllm import LLM, SamplingParams

PSEUDO_ANSWER_PROMPT = GENERATE_PROMPT["pseudo_answer"]

def get_gold_ctxs(supporting_facts, contexts):
    titles = contexts["title"]
    sentences_list = contexts["sentences"]
    gold_titles = supporting_facts["title"]
    gold_sent_ids = supporting_facts["sent_id"]

    grouped_facts = defaultdict(list)
    for i, gold_title in enumerate(gold_titles):
        try:
            actual_idx = titles.index(gold_title)
            target_sent_idx = gold_sent_ids[i]
            gold_sentence = sentences_list[actual_idx][target_sent_idx]
            grouped_facts[gold_title].append(gold_sentence)
        except (ValueError, IndexError) as e:
            # print(f"Warning: Fact parsing error - {e}")
            continue

    gold_ctx = []
    for title, sentences in grouped_facts.items():
        combined_sentences = " ".join(sentences)
        ctx = f"Title: {title}\n\n{combined_sentences}"
        gold_ctx.append(ctx)
    return "\n".join(gold_ctx)

def prepare_prompt_inputs(prompt, tokenizer, device):
    messages = [
        {"role": "user", "content": prompt}
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    return inputs

def generate_pseudo_answers(data, model, tokenizer, sampling_params, batch_size=128):
    valid_indices = []
    
    # device = model.device
    gold_count = 0
    prompts = []
    for i, item in tqdm(enumerate(data), desc="Generating Pseudo Answers"):
        question = item["question"]
        supporting_facts = item["supporting_facts"]
        contexts = item["context"]
        final_short_answer = item["answer"]

        gold_ctx = get_gold_ctxs(supporting_facts, contexts)
        if not gold_ctx:
            # print(f"Warning: No gold context found for question: {question}")
            # item["pseudo_answer"] = None
            # new_data.append(item)
            continue
    
        prompt = PSEUDO_ANSWER_PROMPT.format(
            question=question,
            supporting_facts=gold_ctx,
            final_short_answer=final_short_answer
        )
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, 
            add_generation_prompt=True
        )
        prompts.append(prompt_text)
        gold_count += 1
        valid_indices.append(i)
        # inputs = prepare_prompt_inputs(prompt, tokenizer, device)

    generated_results = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating Pseudo Answers"):
        batch_prompts = prompts[i : i + batch_size]
        outputs = model.generate(batch_prompts, sampling_params, use_tqdm=False)

        for output in outputs:
            generated_text = output.outputs[0].text.strip()
            generated_results.append(generated_text)
    
    new_data = []
    final_answers = [None] * len(data)
    for valid_idx, gen_text in zip(valid_indices, generated_results):
        final_answers[valid_idx] = gen_text
        
    for final_answer, item in zip(final_answers, data):
        item["pseudo_answer"] = final_answer
        new_data.append(item)
    
    print(f"Generated pseudo answers for {gold_count}/{len(data)} examples with valid gold contexts.")
    return new_data


def main():
    argv = sys.argv[1:]
    dataset_name = argv[0]
    _split = argv[1]
    
    filepath = f"data/{dataset_name}/raw/{_split}.jsonl"
    output_path = f"data/{dataset_name}/raw/{_split}_p.jsonl"
    
    if not os.path.exists(filepath):
        print(f"Error: File not found - {filepath}")
        return
    # if filepath == "data/hotpotqa-w/raw/validation.jsonl":
    #     print("Already processed.")
    #     return

    data = load_json_data(filepath)
    model_name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = LLM(model_name_or_path, tensor_parallel_size=2, disable_log_stats=True)
    sampling_params = SamplingParams(
        temperature=0.0, 
        max_tokens=200,
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    )
    # model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", torch_dtype="auto")
    
    new_data = generate_pseudo_answers(data, model, tokenizer, sampling_params)
    with open(output_path, "w") as f:
        for item in new_data:
            f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    main()