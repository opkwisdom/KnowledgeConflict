from omegaconf import DictConfig
from typing import Any, Dict, List, Tuple, Union, Optional
import torch
import logging
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BatchEncoding,
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
        self.hidden_extraction_layers: List[int] = config.model.hidden_extraction_layers

        # Another core components
        self.__post_init__()
    
    def set_base_chat_template(self, task: str = "qa"):
        # Use base template for internal answer generation
        prefix, postfix = template(self.model_name, task, base_template=True)
        self.sys_prompt_ids, self.postfix_ids = self.encode(prefix)[0], self.encode(postfix)[0]
        self.sys_prompt_text = prefix
        self.postfix_text = postfix
    
    def __post_init__(self):
        self.set_base_chat_template()
        self.tokenizer.padding_side = "left"

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
    
    def encode(self, text: Union[str, List[str]], return_tokens_only: bool = True) -> Union[torch.Tensor, BatchEncoding]:
        if isinstance(text, str):
            text = [text]
        
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
        )

        if return_tokens_only:
            return encoded.input_ids.to(self.device)
        return encoded
    
    def decode(self, input_ids: torch.Tensor, is_special_prompt: bool = False) -> List[str]:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        decoded = self.tokenizer.batch_decode(
            input_ids,
            skip_special_tokens=(not is_special_prompt),
        )
        return decoded
    
    def apply_template(self, queries: List[str]) -> torch.Tensor:
        raise NotImplementedError("method `apply_template` is not implemented yet.")


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
        queries: List[str],
        contexts_list: List[List[CtxExample]],   
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """
        Self-Generated Conflict Amplification (SCA) pipeline
        Returns:
            kvformer_input: Tensor of shape (B, L_select, S, D)
            kvformer_mask: Tensor of shape (B, S)
        """
        a_int_list = self.generate_internal_answers(queries)
        
        flat_input_texts = []
        max_target_length = -1
        target_length_list = []
        self_task_prompt_text = self._kvzip.prompt

        # TODO: SCA batch prompt construction
        for i, (q_text, a_int, ctx_list) in enumerate(zip(queries, a_int_list, contexts_list)):
            self_task_prompt = f"{q_text}\n\n{a_int}\n\n{self_task_prompt_text}{self.postfix_text}{a_int}"
            self_task_prompt_ids = self.encode(self_task_prompt)[0]
            
            # Keep track of variable target lengths across different samples
            target_length = len(self_task_prompt_ids)
            target_length_list.append(target_length)
            max_target_length = max(max_target_length, target_length)
            for ctx in ctx_list:
                full_text = f"{self.sys_prompt_text}\n\nTitle: {ctx.title}\n\n{ctx.text}\n\n{self_task_prompt}"
                flat_input_texts.append(full_text)
        
        sca_encoded = self.encode(flat_input_texts, return_tokens_only=False).to(self.device)
        with torch.no_grad():
            outputs = self.model(**sca_encoded, use_cache=False, output_hidden_states=True)

        # Which hidden states to return?
        selected_hiddens = [
            outputs.hidden_states[layer_idx]
            for layer_idx in self.hidden_extraction_layers
        ]
        selected_hiddens = torch.stack(selected_hiddens, dim=1)     # (B, L_select, S, D)
        kvformer_input = selected_hiddens[:, :, -max_target_length:, :]
        kvformer_mask = torch.zeros_like(kvformer_input[:, 0, :, 0], dtype=torch.long).to(self.device)
        
        for i, target_length in enumerate(target_length_list):
            start_idx = i * self.config.data.topk_per_query
            end_idx = start_idx + self.config.data.topk_per_query
            kvformer_mask[start_idx:end_idx, -target_length:] = 1
        
        return kvformer_input, kvformer_mask
    
    @torch.inference_mode()
    def generate_internal_answers(
        self,
        queries: Union[str, List[str]],
    ) -> List[str]:
        """
        Generate internal answers using the base prompt template.
        """
        if isinstance(queries, str):
            queries = [queries]
        
        # Construct input_ids with prompt template
        full_prompts = []
        for query in queries:
            formatted_q = self.base_prompt.format(question=query)
            full_prompt = f"{self.sys_prompt_text}\n\n{formatted_q}{self.postfix_text}"
            full_prompts.append(full_prompt)
        encoded = self.encode(full_prompts, return_tokens_only=False).to(self.device)

        output = self.model.generate(**encoded, **self._kvzip.gen_kwargs)
        input_len = encoded.input_ids.shape[1]
        gen_ids = output[:, input_len:]
        generated_text = self.decode(gen_ids)

        return generated_text