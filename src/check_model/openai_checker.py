from openai import OpenAI
from typing import List
import os


from .llm_checker import LLMChecker, JudgeOutput, CtxsRelevance
from .checker_prompt import OPENAI


class OpenAIChecker(LLMChecker):
    """
    OpenAI LLM Checker implementation
    """
    def __init__(self, llm_model_name: str = "gpt-4o-mini", prompt_name: str = "base"):
        super().__init__(llm_model_name)
        self.client = None
        self.prompt = self.set_prompt(prompt_name)

    def _init_llm(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def set_prompt(self, prompt: str):
        return OPENAI.get(prompt, "")

    def judge(self, query: str, answer: str, contexts: List[str]) -> JudgeOutput:
        judge_output: JudgeOutput = None
        
        # Prepare the input prompt
        input_prompt = self.prompt.format(
            query=query,
            answer=answer,
            contexts="\n".join([f"{i}. {ctx}" for i, ctx in enumerate(contexts)])
        )
        # TODO: Call OpenAI API
        
        return judge_output