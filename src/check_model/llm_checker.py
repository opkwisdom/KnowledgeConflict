import torch
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import os
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

@dataclass
class CtxsRelevance(BaseModel):
    positive: List[int] = Field(..., description="List of 0-based indices for positive contexts")
    negative: List[int] = Field(..., description="List of 0-based indices for negative contexts")
    irrelevant: List[int] = Field(..., description="List of 0-based indices for irrelevant contexts")

@dataclass
class JudgeOutput(BaseModel):
    reasoning: str = Field(..., description="The reasoning provided by the LLM for its judgment")
    is_correct: bool = Field(..., description="Whether the answer is judged correct or not")
    ctx_relevance: CtxsRelevance = Field(..., description="Contextual relevance information")



class LLMChecker(ABC):
    """
    Abstract base class for LLM checkers
    """
    def __init__(self, llm_model_name: str):
        self.llm_model_name = llm_model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_llm()

    @abstractmethod
    def _init_llm(self):
        raise NotImplementedError
    
    @abstractmethod
    def set_prompt(self, prompt: str):
        raise NotImplementedError
    
    @abstractmethod
    def judge(self, query: str, answer: str, contexts: List[str]) -> JudgeOutput:
        raise NotImplementedError