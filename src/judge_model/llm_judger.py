import torch
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import os
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from omegaconf import DictConfig

class CtxsRelevance(BaseModel):
    positive: List[int] = Field(..., description="List of 0-based indices for positive contexts")
    negative: List[int] = Field(..., description="List of 0-based indices for negative contexts")
    # irrelevant: List[int] = Field(..., description="List of 0-based indices for irrelevant contexts")   # Optional

    # @property
    # def mapping(self) -> Dict[int, str]:
    #     mapping = {}
    #     for label, idxs in [
    #         ("positive", self.positive),
    #         ("negative", self.negative),
    #         ("irrelevant", self.irrelevant),
    #     ]:
    #         for idx in idxs:
    #             mapping[idx] = label
        
    #     total_len = len(self.positive) + len(self.negative) + len(self.irrelevant)
    #     for i in range(total_len):
    #         if i not in mapping:
    #             mapping[i] = "irrelevant"
    #     return mapping

    @property
    def mapping(self) -> Dict[int, str]:
        mapping = {}
        for idx in self.positive:
            mapping[idx] = "positive"
        for idx in self.negative:
            mapping[idx] = "negative"
        # irrelevant는 맵핑에 없으면 그냥 없는 것으로 간주하거나,
        # 필요하다면 외부에서 전체 길이를 알 때 처리 (여기서는 생략)
        if 0 not in mapping:
            mapping[0] = "irrelevant"
            
        return mapping

class JudgeOutput(BaseModel):
    is_correct: bool = Field(..., description="Whether the answer is judged correct or not")
    ctx_relevance: CtxsRelevance = Field(..., description="Contextual relevance information")



class LLMJudger(ABC):
    """
    Abstract base class for LLM judgers
    """
    def __init__(self, config: DictConfig):
        self.config = config
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