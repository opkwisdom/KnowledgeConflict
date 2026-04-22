from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils import (
    load_qa_dataset, QAExample
)
import json
import torch
from tqdm import tqdm
from dataclasses import asdict

def single_recon(data: QAExample, model, tokenizer) -> str:
    prefix_text = "Document: "
    suffix_text = " means the same as"
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False, return_tensors="pt").to(model.device)
    suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False, return_tensors="pt").to(model.device)
    prefix_embeds = model.get_input_embeddings()(prefix_ids)
    suffix_embeds = model.get_input_embeddings()(suffix_ids)

    recon_ctxs = []
    ctxs = data.ctxs[:10]  # Limit to top 10 contexts for testing
    for ctx in tqdm(ctxs):
        # Run reconstruction for each context separately
        input_text = f"Title: {ctx.title}\n\n{ctx.text}{tokenizer.eos_token}"
        doc_tokenized = tokenizer(input_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.model(**doc_tokenized, use_cache=False)
            doc_hidden = output.last_hidden_state[:, -1, :]

        combined_embeds = torch.cat([prefix_embeds, doc_hidden.unsqueeze(1), suffix_embeds], dim=1)
        with torch.no_grad():
            gen_outputs = model.generate(
                inputs_embeds=combined_embeds,
                max_new_tokens=50,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        recon_text = tokenizer.decode(gen_outputs[0], skip_special_tokens=True)
        recon_ctxs.append(recon_text.strip())
    return recon_ctxs

def aggregate_recon(data: QAExample, model, tokenizer) -> str:
    prefix_text = "Document: "
    suffix_text = " means the same as"
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False, return_tensors="pt").to(model.device)
    suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False, return_tensors="pt").to(model.device)
    prefix_embeds = model.get_input_embeddings()(prefix_ids)
    suffix_embeds = model.get_input_embeddings()(suffix_ids)

    combined_embeds_list = []
    for ctx in data.ctxs:
        input_text = f"Title: {ctx.title}\n\n{ctx.text}{tokenizer.eos_token}"
        doc_tokenized = tokenizer(input_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.model(**doc_tokenized, use_cache=False)
            doc_hidden = output.last_hidden_state[:, -1, :]
        combined_embeds_list.append(doc_hidden)
    combined_doc_hidden = torch.stack(combined_embeds_list, dim=1).mean(dim=1)
    
    combined_embeds = torch.cat([prefix_embeds, combined_doc_hidden.unsqueeze(1), suffix_embeds], dim=1)
    with torch.no_grad():
        gen_outputs = model.generate(
            inputs_embeds=combined_embeds,
            max_new_tokens=50,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    recon_text = tokenizer.decode(gen_outputs[0], skip_special_tokens=True)
    return recon_text.strip()


def main():
    data = load_qa_dataset("data/hotpotqa-w/retrieved/validation_top100.jsonl")
    data = data[0]
    model_name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", torch_dtype="auto")

    single_recon_result = single_recon(data, model, tokenizer)
    aggregate_recon_result = aggregate_recon(data, model, tokenizer)

    result = {
        "question": data.question,
        "answers": data.answers,
        "ctxs": [asdict(ctx) for ctx in data.ctxs],
        "single_recon": single_recon_result,
        "aggregate_recon": aggregate_recon_result
    }
    with open("recon_test_result.json", "w") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    main()