from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List

from .llm_checker import LLMChecker, JudgeOutput, CtxsRelevance

class HfLLMChecker(LLMChecker):
    """
    HuggingFace LLM Checker implementation
    """
    def __init__(self, llm_model_name: str):
        super().__init__(llm_model_name)
        self.model = None
        self.tokenizer = None
        self.prompt = None
        self.prefix = ""
        self.postfix = ""

    def _init_llm(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.llm_model_name)
        self.model.eval()
        self.model.to(self.device)

    def set_prompt(self, prompt: str):
        self.prompt = prompt

    def set_template(self, prefix: str, postfix: str):
        self.prefix = prefix
        self.postfix = postfix
    
    def judge(self, query: str, answer: str, contexts: List[str]) -> JudgeOutput:
        raise NotImplementedError