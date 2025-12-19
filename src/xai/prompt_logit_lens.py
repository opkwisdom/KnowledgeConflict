from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import glob
import torch
import re

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto").to("cuda:0")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
tokenizer.pad_token_id = tokenizer.eos_token_id

TARGET_IDX = tokenizer.eos_token_id
INSPECT_DIR = "/workspaces/kvzip_nlplab/KFC-scoring/prompt_analysis_dataset/20251213_072621"

SYS_PROMPT = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
SYS_PROMPT += "You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
POSTFIX = "\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


def parse_prompt(inspect_dir: str) -> str:
    """
    Parse the base prompt used during inference from the log file
    """
    log_files = glob.glob(f"{inspect_dir}/*.log")
    if not log_files:
        print(f"Error: No .log file found in {inspect_dir}")
        return None

    log_file = log_files[0]
    with open(log_file, "r") as f:
        log_text = f.readlines()
    
    base_prompt = []
    parsing_mode = False

    LOG_PATTERN = re.compile(r"^\s*\[\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}\s-\s.*?\s-\s(INFO|DEBUG|WARNING|ERROR|CRITICAL)\]")
    PROMPT_START_KEY = "Using base prompt: "

    for line in log_text:
        if PROMPT_START_KEY in line and not parsing_mode:
            try:
                content_start_idx = line.index(PROMPT_START_KEY) + len(PROMPT_START_KEY)
                prompt_line = line[content_start_idx:].rstrip()
                base_prompt.append(prompt_line)
                parsing_mode = True
            except ValueError:
                continue
        elif parsing_mode:
            if LOG_PATTERN.match(line):
                parsing_mode = False
                break
            base_prompt.append(line.rstrip())
    
    if base_prompt:
        return "\n".join(base_prompt)
    else:
        print("Error: No prompt found in the log file.")
        return None

def construct_teacher_forcing_input(
    base_prompt: str,
    question: str,
    generated_text: str,
    tokenizer
):
    prompt_filled = base_prompt.format(question=question)
    input_text = f"{SYS_PROMPT}{prompt_filled}{POSTFIX}"
    input_ids = tokenizer.encode(
        input_text,
        return_tensors="pt",
        add_special_tokens=False,
    ).to("cuda:0")

    input_len = input_ids.size(1)
    gen_ids = tokenizer.encode(
        generated_text,
        return_tensors="pt",
        add_special_tokens=False,
    ).to("cuda:0")

    full_input_ids = torch.cat([input_ids, gen_ids], dim=1)
    return full_input_ids, input_len

def analyze_logit_lens(
    lm_head: torch.nn.Linear,
    target_hidden_states: torch.Tensor,
    target_token_id: int,
) -> torch.Tensor:
    L, S, D = target_hidden_states.shape
    all_probs = []
    for layer in range(L):
        layer_hidden_states = target_hidden_states[layer]  # (S, D)
        with torch.no_grad():
            layer_logits = lm_head(layer_hidden_states).to("cpu")  # (S, Vocab)
        # layer_probs = torch.softmax(layer_logits, dim=-1)
        target_probs = layer_logits[:, target_token_id]  # (S,)
        all_probs.append(target_probs)
    all_probs = torch.stack(all_probs)  # (L, S)
    return all_probs

def eos_sensitivity_score(
    sample_logits: torch.Tensor,
    ref_len: int,
    alpha: float = 1.2
) -> float:
    mean_logit = sample_logits.mean().item()
    min_logit = sample_logits.min().item()
    internal_score = mean_logit * min_logit

    gen_len = sample_logits.size(1)

    allowed_len = ref_len * alpha
    if gen_len <= allowed_len:
        length_penalty = 1.0
    else:
        length_penalty = allowed_len / gen_len
    
    final_score = internal_score * length_penalty
    return final_score
    

### Get test dataset
with open(f"{INSPECT_DIR}/inference_results.json", "r") as f:
    test_dataset = json.load(f)

### Parse base prompt
base_prompt = parse_prompt(INSPECT_DIR)
print(base_prompt)

lm_head = model.get_output_embeddings()

test_data = test_dataset["param_true"]
for item in test_data:
    question = item["question"]
    generated_text = item["pred_answer"]
    ref_len = max([len(ans) for ans in item["answers"]])

    full_input_ids, input_len = construct_teacher_forcing_input(
        base_prompt,
        question,
        generated_text,
        tokenizer
    )
    # Teacher forcing to get hidden states
    with torch.no_grad():
        outputs = model(full_input_ids, output_hidden_states=True)

    target_hidden_states = torch.stack([
        hs[0, input_len-1:, :] for hs in outputs.hidden_states
    ])  # (layer, gen_len, hidden_size)
    sample_logits = analyze_logit_lens(lm_head, target_hidden_states, TARGET_IDX)
    # Compute EOS sensitivity score
    eos_sens_score = eos_sensitivity_score(sample_logits, ref_len)
    print(f"Question: {question}")
    print(f"Reference Length: {ref_len}")
    print(f"EOS Sensitivity Score: {eos_sens_score}\n")
    # print(sample_logits)