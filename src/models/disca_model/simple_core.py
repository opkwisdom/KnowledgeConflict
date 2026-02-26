from omegaconf import DictConfig
from typing import Any, Dict, List, Tuple, Union, Optional
import torch
import logging
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    LlamaConfig,
)
import json

from KVzip.model import ModelKVzip
from KVzip.attention import RetainCache, EvictCache
from utils import CtxExample, CtxsRelevance, template

logger = logging.getLogger(__name__)


class DISCA:
    """
    Decoding Intervention through Self-Generated Conflict Amplification framework.
    """
    def __init__(self, config: DictConfig, kvzip: ModelKVzip, generate_prompt: str, base_prompt: str) -> None:
        self.config: DictConfig = config
        self._kvzip: ModelKVzip = kvzip
        self.generate_prompt: str = generate_prompt
        self.base_prompt: str = base_prompt
        self.model_name: str = config.model.model_name

        # Another core components
        self.__post_init__()
    
    def set_base_chat_template(self, task: str = "qa"):
        # Use base template for internal answer generation
        prefix, postfix = template(self.model_name, task, base_template=True)
        self.sys_prompt_ids, self.postfix_ids = self._kvzip.encode(prefix), self._kvzip.encode(postfix)
    
    def __post_init__(self):
        self.set_base_chat_template()

    # For user-convenient access
    @property
    def device(self) -> torch.device:
        return self._kvzip.device
    
    @property
    def model(self) -> AutoModelForCausalLM:
        return self._kvzip.model
    
    @property
    def tokenizer(self) -> AutoTokenizer:
        return self._kvzip.tokenizer

    @torch.inference_mode()
    def prefill(
        self,
        ctx_ids: Union[str, torch.Tensor],
        q_ids: torch.Tensor,
        a_ids: Optional[torch.Tensor] = None,
        prefill_chunk_size: int = 16000,
        load_score=False,
        do_score=True,
    ) -> Union[RetainCache, EvictCache]:
        # Use KVzip prefill method
        return self._kvzip.prefill(
            ctx_ids,
            q_ids=q_ids,
            a_ids=a_ids,
            prefill_chunk_size=prefill_chunk_size,
            load_score=load_score,
            do_score=do_score,
        )
    
    def __call__(
        self,
        query: str,
        contexts: List[CtxExample],   
    ) -> torch.Tensor:
        """
        Self-Generated Conflict Amplification (SCA) pipeline
        """
        a_internal = self.generate_internal_answer(query)
        sca_prompts = [
            f"Title: {ctx_ex.title}\n\n{ctx_ex.text}"
            for ctx_ex in contexts
        ]

        q_ids = self._kvzip.encode(query) if isinstance(query, str) else query
        a_ids = self._kvzip.encode(a_internal) if isinstance(a_internal, str) else a_internal
        sca_prompts_ids = self._kvzip.encode(sca_prompts)

        _, hidden_states = self.prefill(
            ctx_ids=sca_prompts_ids,
            q_ids=q_ids,
            a_ids=a_ids
        )
        return hidden_states
    
    @torch.inference_mode()
    def generate_internal_answer(
        self,
        query: str,
    ) -> str:
        # Construct input_ids with prompt template
        input_text = self.base_prompt.format(question=query)
        input_ids = self._kvzip.apply_template(input_text)
        input_ids = torch.cat([self.sys_prompt_ids, input_ids], dim=1)
        input_ids = input_ids.to(self.device)

        output = self.model.generate(input_ids, **self._kvzip.gen_kwargs)
        gen_ids = output[:, len(input_ids[0]):-1]
        generated_text = self._kvzip.decode(gen_ids)

        return generated_text