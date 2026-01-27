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
from src.utils import CtxExample, CtxsRelevance, BoostedProbResult, template, compute_metrics

from .conflict_resources import *
from .conflict_handler import ConflictConfigHandler
from .lexical_cue import LexicalCueEmbedder
from .uncertainty_estimator import UncertaintyEstimator

class KnowledgeFusionCore:
    def __init__(self, config: DictConfig, kvzip: ModelKVzip, generate_prompt: str, base_prompt: str, logger: logging.Logger) -> None:
        self.config: DictConfig = config
        self._kvzip: ModelKVzip = kvzip
        self.generate_prompt: str = generate_prompt
        self.base_prompt: str = base_prompt
        self.logger: logging.Logger = logger
        self.model_name: str = config.model.model_name

        # Another core components
        self.conflict_handler = ConflictConfigHandler(config, kvzip.model.config)
        # self.lex_cue_embedder = LexicalCueEmbedder(config.model.lexical_cue, self.conflict_handler)
        # self.uncertainty_estimator = UncertaintyEstimator(config.uncertainty_estimator)
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
        return_idx: bool = False,
    ) -> Union[List[EvictCache], List[Tuple[int, EvictCache]]]:
        evicted_kvs = []

        for ctx_idx, ctx_ex in enumerate(contexts):
            context = f"Title: {ctx_ex.title}\n\n{ctx_ex.text}"
            try:
                relevance = relevance_map[ctx_idx]
            except KeyError:
                self.logger.error(f"Context index {ctx_idx} not found in relevance map.")
                relevance = "irrelevant"

            # Case 1 - Irrelevant context
            if relevance == "irrelevant":
                continue
            # Case 2 - Conflict context
            else:
                prune_map = self.conflict_handler.critical_map if relevance == "negative"\
                                else self.conflict_handler.normal_map
                kv = self.prefill(
                    ctx_ids=context,
                    q_ids=q_ids,
                    a_ids=a_ids,
                )
                kv.prune(
                    ratio=prune_ratio,
                    prune_kwargs=prune_map,
                    prune_type=relevance
                )
                # evicted_kvs.append((relevance, kv))
                evicted_kvs.append((ctx_idx, kv) if return_idx else kv)
        
        return evicted_kvs
    
    @torch.inference_mode()
    def resolve_and_generate(
        self,
        query: Union[str, torch.Tensor],
        contexts: List[CtxExample],
        relevance: CtxsRelevance,
        internal_answer: str,
        use_single_context: bool = True,
    ) -> Tuple[str, str]:
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
        # Lexical cue embedding
        # all_kv = self.lex_cue_embedder.embed_lexical_cues(tagged_all_kv)
        # if isinstance(tagged_all_kv[0], tuple):
            # all_kv = []
            # for tagged_kv in tagged_all_kv:
                # all_kv.append(tagged_kv[1])

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

        if kv is None:  # Irrelevant case
            generated_text = internal_answer
            return generated_text, final_rel_type

        # Generate
        # To generate correctly, need to apply template
        input_text = self.generate_prompt.format(question=query)
        input_ids = self._kvzip.apply_template(input_text)
        input_ids = input_ids.to(self.device)

        if kv.prefill_ids is not None:  # prefill_ids has already involved system prompt
            input_ids = torch.cat([kv.prefill_ids, input_ids], dim=1)
        output = self.model.generate(input_ids, past_key_values=kv, **self._kvzip.gen_kwargs)
        gen_ids = output[:, len(input_ids[0]):-1]  # parse response
        generated_text = self._kvzip.decode(gen_ids)
        
        return generated_text, final_rel_type
    
    def filter_irrelevant_contexts(
        self,
        question: str,
        internal_answer: str,
        contexts: List[CtxExample]
    ) -> Tuple[List[CtxExample], List[CtxExample]]:
        relevant_contexts = []
        irrelevant_contexts = []

        for ctx_ex in contexts:
            context = f"Title: {ctx_ex.title}\n\n{ctx_ex.text}"
            kv = self.prefill(
                ctx_ids=context,
                q_ids=self._kvzip.encode(f"Question: {question}\n"),
                a_ids=self._kvzip.encode(f"Answer: {internal_answer}\n"),
                load_score=False,
                do_score=True,
            )
            kv.to("cpu")

            is_rel, _ = kv.validate_relevance(
                topk=self.config.model.conflict_topk,
                control_cache_stats=self.conflict_handler.control_cache_stats,
                control_d=self.conflict_handler.selected_d
            )
            if is_rel:
                relevant_contexts.append(ctx_ex)
            else:
                irrelevant_contexts.append(ctx_ex)
        
        return relevant_contexts, irrelevant_contexts
    
    @torch.inference_mode()
    def generate_internal_answer(
        self,
        query: Union[str, torch.Tensor],
        output_attentions: bool = False,
    ) -> Union[str, Tuple[str, float], Tuple[str, BoostedProbResult]]:
        # Construct input_ids with prompt template
        input_text = self.base_prompt.format(question=query)
        input_ids = self._kvzip.apply_template(input_text)
        input_ids = torch.cat([self.sys_prompt_ids, input_ids], dim=1)
        input_ids = input_ids.to(self.device)

        # BoostedProb or not
        gen_config = self._kvzip.gen_kwargs.copy()
        if output_attentions:
            gen_config.update({
                "output_attentions": True,
                "return_dict_in_generate": True,
            })
        output = self.model.generate(input_ids, **gen_config)

        # Quality estimation
        if not isinstance(output, str):
            sequences = output.sequences
            import pdb; pdb.set_trace()
            # boosted_prob_result = self.uncertainty_estimator.calibrate_inspect(output.scores)\
            #     if inspect_mode else self.uncertainty_estimator.calibrate(output.scores)
        else:
            sequences = output

        gen_ids = sequences[:, len(input_ids[0]):-1]
        generated_text = self._kvzip.decode(gen_ids)

        if output_attentions:
            return generated_text, None

        return generated_text
    
    @torch.inference_mode()
    def extract_features(
        self,
        query: Union[str, torch.Tensor],
        contexts: List[CtxExample],
        relevance: CtxsRelevance,
        internal_answer: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features from evicted KV caches to train conflict detector.
        """
        # input query & internal answer
        q_ids = self._kvzip.encode(query) if isinstance(query, str) else query
        a_ids = self._kvzip.encode(internal_answer) if isinstance(internal_answer, str) else internal_answer
        
        relevance_map = relevance.mapping
        tagged_all_kv = self.get_evicted_kvs(
            contexts,
            q_ids=q_ids,
            a_ids=a_ids,
            relevance_map=relevance_map,
            prune_ratio=self.config.model.prune.ratio,
            return_idx=True,
        )

        features = {}
        for idx, kv in tagged_all_kv:
            cross_attention_scores = torch.cat(kv.score, dim=0)
            max_features = torch.amax(cross_attention_scores, dim=-1)
            entropy_features = -torch.sum(
                cross_attention_scores * torch.log(cross_attention_scores + 1e-10),
                dim=-1
            )
            sum_features = torch.sum(cross_attention_scores, dim=-1)

            relevance_type = relevance_map[idx]
            features[relevance_type] = torch.stack([
                max_features,
                entropy_features,
                sum_features,
            ], dim=-1).cpu()  # (layer, head, 3)

        return features
    # Max, Entropy, Sum features, Delta features