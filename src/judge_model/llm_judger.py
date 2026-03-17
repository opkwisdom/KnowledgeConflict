import torch
from abc import ABC, abstractmethod
from typing import List, Dict, Literal
from pydantic import BaseModel, Field
from omegaconf import DictConfig

from utils import CtxsRelevance


class ContextEvaluation(BaseModel):
    index: int = Field(..., description="The index of the context being evaluated (e.g., 0, 1, 2).")
    reasoning: str = Field(..., description="A single, concise reason why this context is S, C, or I in max 20 words.")
    category: Literal["S", "C", "I"] = Field(..., description="The strict category classification.")

class CtxsRelevanceParser(BaseModel):
    evaluations: List[ContextEvaluation] = Field(..., description="A complete list of evaluations for EVERY context provided. You must include all indices.")
    
    def to_dataclass(self) -> "CtxsRelevance":
        sup, ctd, irr = [], [], []
        for eval in self.evaluations:
            if eval.category == "S":
                sup.append(eval.index)
            elif eval.category == "C":
                ctd.append(eval.index)
            elif eval.category == "I":
                irr.append(eval.index)

        return CtxsRelevance(
            supportive=sup,
            contradictory=ctd,
            irrelevant=irr
        )


class JudgeOutput(BaseModel):
    ctx_relevance: CtxsRelevance = Field(..., description="Contextual relevance information")



class LLMJudger(ABC):
    """
    Abstract base class for LLM judgers
    """
    def __init__(self, config: DictConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else -1
        self._init_llm()

    @abstractmethod
    def _init_llm(self):
        raise NotImplementedError
    
    @abstractmethod
    def set_prompt(self, prompt: str):
        raise NotImplementedError
    
    @abstractmethod
    def judge(self, query: str, answer: List[str], contexts: List[str]) -> JudgeOutput:
        raise NotImplementedError