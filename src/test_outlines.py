import outlines
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict
from pydantic import BaseModel, Field
from tqdm import tqdm
import json


def template(model_name, task="qa", system_prompt="", base_template=False):
    model_name = model_name.lower()

    if "llama" in model_name or model_name == "duo":
        # borrowed from https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1
        prefix = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        prefix += "You are a helpful assistant. {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"

        postfix = "\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    elif model_name.startswith("qwen"):
        # https://github.com/QwenLM/Qwen3
        prefix = "<|im_start|>system\nYou are a helpful assistant. {system_prompt}<|im_end|>\n"
        prefix += "<|im_start|>user\n"

        postfix = "<|im_end|>\n<|im_start|>assistant\n"
        if "qwen3-" in model_name:
            postfix += "<think>\n\n</think>\n\n"

    elif model_name.startswith("gemma3") or model_name.startswith("gemma-3"):
        prefix = "<bos><start_of_turn>user\n"
        prefix += "You are a helpful assistant. {system_prompt}\n\n"

        postfix = "<end_of_turn>\n<start_of_turn>model\n"

    else:
        print("**Warning** The model template does not exist! Check data/template.py")
        prefix = "<|begin_of_text|>"
        postfix = "\n\nAnswer: "

    if base_template:
        return prefix.format(system_prompt=system_prompt), postfix

    if task.startswith("gsm"):
        prefix += "Given the context, answer to the following reasoning question.\n\n"
    else:
        prefix += "Given the context, answer to the following question or request without explanation.\n\n"

    return prefix.format(system_prompt=system_prompt), postfix

def apply_template(query, context, model_name, task="qa", system_prompt="", base_template=False):
    prefix, postfix = template(model_name, task=task, base_template=base_template, system_prompt=system_prompt)
    query = f"\n\n{query.strip()}"
    if context is not None:
        query = f"{context.strip()}\n\n{query}"
    return f"{prefix}{query}{postfix}"


HUGGINGFACE = {
    "base": {
        # template.py가 "You are a helpful assistant." 뒤에 붙일 내용
        "system": (
            "You are an extremely strict evaluator for a RAG system. "
            "Your goal is to evaluate context relevance and answer correctness based on strict guidelines."
        ),
        
        # apply_template의 'query' 인자로 들어갈 내용
        "user": (
            "### Task Description\n"
            "1. **Reasoning**: Explain your logic briefly.\n"
            "2. **Analyze Contexts**: Classify EVERY context (from [0] to [{last_index}]) into one of three categories based on these rules"
            "DO NOT skip any context:\n\n"
            "   - **Positive**: Contains specific, literal evidence for the answer. No deduction allowed.\n"
            "   - **Negative**: Topic/keywords match but FAILS to answer (Hard Negative/Distractor).\n"
            "   - **Irrelevant**: Off-topic or unrelated.\n"
            "3. **Determine Correctness**: Check if the Internal Answer provides a factually correct response to the Query.\n\n"

            "### Input Data\n"
            "- Query: {query}\n"
            "- Internal Answer: {internal_answer}\n\n"
            
            "### Contexts\n"
            "{formatted_contexts}\n\n"
            
            "Analyze the data and provide the structured evaluation."
        )
    }
}

HUGGINGFACE = {
    "base": {
        # template.py가 "You are a helpful assistant." 뒤에 붙일 내용
        "system": (
            "You are an extremely strict evaluator for a RAG system. "
            "Your goal is to evaluate context relevance and answer correctness based on strict guidelines."
        ),
        
        # apply_template의 'query' 인자로 들어갈 내용
        "user": (
            "### Task Description\n"
            "1. **Reasoning**: Explain your logic briefly.\n"
            "2. **Analyze Contexts**: Classify EVERY context (from [0] to [{last_index}]) into one of three categories based on these rules"
            "DO NOT skip any context:\n\n"
            "   - **Positive**: The context contains the specific information to derive the answer literally."
            "Deduction or general knowledge is NOT allowed. The evidence must be present in the text.\n"
            "   - **Negative**: Classify as Negative if the context shares ANY semantic relationship, keywords, or topic with the Query, "
            "but it FAILS to provide the answer. This includes 'hard negatives', 'partial information' or 'outdated facts'.\n"
            "   - **Irrelevant**: The context is unrelated to the Query, discusses a different entity, or is completely off-topic.\n\n"
            "3. **Determine Correctness**: Check if the Internal Answer provides a factually correct based on the Query.\n\n"

            "### Input Data\n"
            "- Query: {query}\n"
            "- Internal Answer: {internal_answer}\n\n"
            
            "### Contexts\n"
            "{formatted_contexts}\n\n"
            
            "Analyze the data and provide the structured evaluation."
        )
    }
}


@dataclass
class CtxsRelevance(BaseModel):
    positive: List[int] = Field(..., description="List of 0-based indices for positive contexts")
    negative: List[int] = Field(..., description="List of 0-based indices for negative contexts")
    irrelevant: List[int] = Field(..., description="List of 0-based indices for irrelevant contexts")

    @property
    def mapping(self) -> Dict[int, str]:
        mapping = {}
        for label, idxs in [
            ("positive", self.positive),
            ("negative", self.negative),
            ("irrelevant", self.irrelevant),
        ]:
            for idx in idxs:
                mapping[idx] = label
        
        total_len = len(self.positive) + len(self.negative) + len(self.irrelevant)
        for i in range(total_len):
            if i not in mapping:
                mapping[i] = "irrelevant"
        return mapping

@dataclass
class JudgeOutput(BaseModel):
    is_correct: bool = Field(..., description="Whether the answer is judged correct or not")
    ctx_relevance: CtxsRelevance = Field(..., description="Contextual relevance information")



MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token_id = tokenizer.eos_token_id

model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("cuda"),
    tokenizer
)

with open("/home/nlplab/work/KFC/data/nq/parametric_relevance_tagged/validation.json") as f:
    ex_data = json.load(f)
ex_data_list = ex_data[:10]

base_prompt = HUGGINGFACE["base"]

system_prompt = base_prompt["system"]
user_prompt = base_prompt["user"]

judge_outputs = []

def sanitize_output(output: JudgeOutput, max_idx: int) -> JudgeOutput:
    """
    LLM이 생성한 인덱스 중 범위를 벗어난(max_idx 초과) 쓰레기 값을 제거
    """
    def filter_idxs(idxs):
        return [i for i in idxs if 0 <= i <= max_idx]

    output.ctx_relevance.positive = filter_idxs(output.ctx_relevance.positive)
    output.ctx_relevance.negative = filter_idxs(output.ctx_relevance.negative)
    output.ctx_relevance.irrelevant = filter_idxs(output.ctx_relevance.irrelevant)
    
    return output

for ex_data in tqdm(ex_data_list):
    user_content = user_prompt.format(
        query=ex_data["question"],
        internal_answer=ex_data["parametric_answer"],
        formatted_contexts="\n".join([f"[{i}]\nTitle: {ctx['title']}\n\n{ctx['text']}\n" for i, ctx in enumerate(ex_data["ctxs"])]),
        last_index=len(ex_data["ctxs"])-1
    )

    prompt = apply_template(
        query=user_content,
        context=None,
        model_name=MODEL_NAME,
        task="qa",
        system_prompt=system_prompt,
        base_template=True,
    )

    output = model(
        prompt,
        JudgeOutput,
        max_new_tokens=128,
    )

    judge_output = JudgeOutput.model_validate_json(output)
    judge_output = sanitize_output(judge_output, max_idx=len(ex_data["ctxs"])-1)
    
    print(f"Is Correct: {judge_output.is_correct}")           
    print(f"Context Relevance: {judge_output.ctx_relevance.mapping}")

    judge_outputs.append(judge_output)

with open("test_output/hf_judge_results.json", "w") as f:
    json.dump([jo.model_dump() for jo in judge_outputs], f, indent=4)