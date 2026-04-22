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

QUESTION_TEMPLATE = "Question: {question}\n\n"
PROMPT_TEMPLATE = "Context:\n{ctx}\n\nQuestion: {question}\n\n"

def calculate_gen_loss(ctxs, question, answers, model, tokenizer):
    prompt_ids_list = []
    full_ids_list = []
    loss_list = []

    if ctxs is None:
        prompt = QUESTION_TEMPLATE.format(question=question)
        messages = [
            {"role": "user", "content": prompt}
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        full_text = prompt_text + answers + tokenizer.eos_token

        prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
        full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(model.device)

        prompt_ids_list.append(prompt_ids)
        full_ids_list.append(full_ids)
    else:
        for ctx in ctxs:
            prompt = PROMPT_TEMPLATE.format(ctx=ctx, question=question)
            messages = [
                {"role": "user", "content": prompt}
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            full_text = prompt_text + answers + tokenizer.eos_token

            prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
            full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(model.device)

            prompt_ids_list.append(prompt_ids)
            full_ids_list.append(full_ids)
    
    for prompt_ids, full_ids in zip(prompt_ids_list, full_ids_list):
        with torch.no_grad():
            prompt_len = prompt_ids.shape[1]
            labels = full_ids.clone()
            labels[:, :prompt_len] = -100
            outputs = model(input_ids=full_ids, labels=labels)
            loss = outputs.loss.item()
            loss_list.append(loss)
    
    return loss_list


def prepare_ctxs(supporting_facts, ctxs):
    gold_ctxs, neg_ctxs = [], []
    titles = ctxs["title"]
    sentences_list = ctxs["sentences"]
    gold_titles = supporting_facts["title"]
    gold_sent_ids = supporting_facts["sent_id"]
    
    for title, sentences in zip(titles, sentences_list):
        sentence_text = " ".join(sentences)
        ctx = f"Title: {title}\n\n{sentence_text}"
        
        if title in gold_titles:
            gold_ctxs.append(ctx)
        else:
            neg_ctxs.append(ctx)
            
    return gold_ctxs, neg_ctxs

def main():
    test_type = sys.argv[1]
    data = load_json_data("data/hotpotqa-w/raw/validation_p.jsonl")
    data = data[:1000]   # Limit to first 1000 examples for testing
    model_name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", torch_dtype="auto")

    results = []
    for item in tqdm(data, desc="Calculating Gen Loss"):
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
        base_loss = calculate_gen_loss(None, question, answers, model, tokenizer)
        gold_loss_list = calculate_gen_loss(gold_ctxs, question, answers, model, tokenizer)
        neg_loss_list = calculate_gen_loss(neg_ctxs, question, answers, model, tokenizer)

        results.append({
            "question": question,
            "answers": answers,
            "base_loss": base_loss,
            "gold_loss": gold_loss_list,
            "neg_loss": neg_loss_list,
        })
    print(f"Calculated generation losses for {len(results)} examples.")
    
    with open(f"src/tests/oracle_loss/{test_type}_genloss_test_result.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()