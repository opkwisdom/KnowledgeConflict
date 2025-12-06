from omegaconf import DictConfig
from typing import Any, Dict, List, Tuple, Union, Optional
import torch
import logging
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    LlamaConfig,
)

from KVzip.model import ModelKVzip
from KVzip.attention import RetainCache, EvictCache
from utils import CtxExample, CtxsRelevance, template
from .conflict_resources import *

class KnowledgeFusionCore:
    def __init__(self, config: DictConfig, kvzip: ModelKVzip, generate_prompt: str, base_prompt: str, logger: logging.Logger) -> None:
        self.config: DictConfig = config
        self._kvzip: ModelKVzip = kvzip
        self.generate_prompt: str = generate_prompt
        self.base_prompt: str = base_prompt
        self.logger: logging.Logger = logger
        self.model_name: str = config.model.model_name
        self.__post_init__()
    
    def set_base_chat_template(self, task: str = "qa"):
        # Use base template for internal answer generation
        prefix, postfix = template(self.model_name, task, base_template=True)
        self.sys_prompt_ids, self.postfix_ids = self._kvzip.encode(prefix), self._kvzip.encode(postfix)

    def _parse_critical_indices(self, critical_indices: Optional[List[str]]
        ) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
        NUM_HEADS = getattr(self.model.config, "num_key_value_heads", None)
        NUM_LAYERS = self.model.config.num_hidden_layers

        critical_map: Dict[int, List[int]] = {}
        normal_map: Dict[int, List[int]] = {}

        for item in critical_indices:
            parts = item.split('_')
            layer_idx = int(parts[0])

            if layer_idx not in critical_map:
                critical_map[layer_idx] = []
            
            if len(parts) == 1:
                critical_map[layer_idx] = list(range(NUM_HEADS))
            else:
                head_idx = int(parts[1])
                critical_map[layer_idx].append(head_idx)
        
        for layer_idx in range(NUM_LAYERS):
            all_heads_in_layer = set(range(NUM_HEADS))
            critical_heads = critical_map.get(layer_idx, set())

            normal_heads = all_heads_in_layer - set(critical_heads)
            if normal_heads:
                normal_map[layer_idx] = list(normal_heads)
        
        return critical_map, normal_map
    
    def __post_init__(self):
        # init conflict
        critical_indices = None
        if isinstance(self.model.config, LlamaConfig):
            critical_indices = LLAMA_CRITICAL_INDICES
        
        if critical_indices:
            self.critical_map, self.normal_map = self._parse_critical_indices(critical_indices)
        else:
            self.critical_map, self.normal_map = {}, {}
            self.logger.warning("No critical indices found for the model.")
        
        self.set_base_chat_template()

    # For user convenience access
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
    
    def get_evicted_kvs(
        self,
        contexts: List[CtxExample],
        q_ids: torch.Tensor,
        a_ids: torch.Tensor,
        relevance_map: CtxsRelevance,
        prune_ratio: float,
    ) -> List[CtxExample]:
        evicted_kvs = []

        for ctx_idx, ctx_ex in enumerate(contexts):
            context = f"Title: {ctx_ex.title}\n\n{ctx_ex.text}"
            try:
                relevance = relevance_map[ctx_idx]
            except KeyError:
                self.logger.error(f"Context index {ctx_idx} not found in relevance map.")
                relevance = "irrelevant"

            # Case 2a - Context is relevant
            if relevance == "positive":
                kv = self.prefill(
                    ctx_ids=context,
                    q_ids=q_ids,
                    a_ids=a_ids,
                )
                kv.prune(
                    ratio=prune_ratio,
                    prune_kwargs=self.normal_map,
                    prune_type=relevance
                )
                evicted_kvs.append(kv)
            
            # Case 2b - Context is negative
            # TODO: Embed [THINK] token logic here
            elif relevance == "negative":
                kv = self.prefill(
                    ctx_ids=context,
                    q_ids=q_ids,
                    a_ids=a_ids,
                )
                kv.prune(
                    ratio=prune_ratio,
                    prune_kwargs=self.critical_map,
                    prune_type=relevance
                )
                evicted_kvs.append(kv)
            # Case 2c - Context is irrelevant
            else:
                kv = None    # Skip
                evicted_kvs.append(kv)
        
        return evicted_kvs
    
    @torch.inference_mode()
    def resolve_and_generate(
        self,
        query: Union[str, torch.Tensor],
        contexts: List[CtxExample],
        relevance: CtxsRelevance,
        internal_answer: str,
        use_single_context: bool = True,
    ) -> Tuple[str, str, List[EvictCache]]:
        # input query & internal answer
        q_ids = self._kvzip.encode(query) if isinstance(query, str) else query
        a_ids = self._kvzip.encode(internal_answer) if isinstance(internal_answer, str) else internal_answer
        
        relevance_map = relevance.mapping
        all_kv = self.get_evicted_kvs(
            contexts,
            q_ids=q_ids,
            a_ids=a_ids,
            relevance_map=relevance_map,
            prune_ratio=self.config.model.prune.ratio,
        )
        
        # KV cache merge strategy
        # Use only single context (temporary)
        merged_kv = None
        final_rel_type = "multiple"
        
        if not all_kv:
            merged_kv = None
            final_rel_type = "irrelevant"
        elif use_single_context:
            merged_kv = all_kv[0]
            final_rel_type = relevance_map[0]
        else:
            merged_kv = EvictCache.merge(all_kv)
            final_rel_type = "multiple"

        kv = self._kvzip._init_kv(kv=merged_kv)
        seen_token_prev = kv._seen_tokens

        # Generate
        # To generate correctly, need to apply template
        input_text = self.generate_prompt.format(question=query)
        input_ids = self._kvzip.apply_template(input_text)
        input_ids = input_ids.to(self.device)

        if kv is not None and kv.prefill_ids is not None:  # prefill_ids has already involved system prompt
            input_ids = torch.cat([kv.prefill_ids, input_ids], dim=1)
        output = self.model.generate(input_ids, past_key_values=kv, **self._kvzip.gen_kwargs)
        gen_ids = output[:, len(input_ids[0]):-1]  # parse response
        generated_text = self._kvzip.decode(gen_ids)
        
        return generated_text, final_rel_type, all_kv
    
    @torch.inference_mode()
    def generate_internal_answer(
        self,
        query: Union[str, torch.Tensor],
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