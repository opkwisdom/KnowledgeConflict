import torch
import logging
from omegaconf import DictConfig
from typing import List, Tuple, Dict, Callable

from KVzip.attention import EvictCache
from .conflict_handler import ConflictConfigHandler

logger = logging.getLogger(__name__)


def log_prob_based_estimation(kv_cache: EvictCache, n_tokens: int) -> torch.Tensor:
    noun_probs = kv_cache.log_probs.masked_fill(~kv_cache.noun_mask, float('-inf'))
    _, lex_cue_positions = torch.topk(noun_probs, k=n_tokens, dim=-1)
    return lex_cue_positions

def max_norm_based_estimation(kv_cache: EvictCache, n_tokens: int) -> torch.Tensor:
    import pdb; pdb.set_trace()
    x=1

def induction_head_based_estimation(kv_cache: EvictCache, n_tokens: int,
                                    induction_head_indices: List[Tuple[int, int]]) -> torch.Tensor:
    import pdb; pdb.set_trace()
    x=1

EST_STRATEGY_FUNCS: Dict[str, Callable] = {
    "log_prob": log_prob_based_estimation,
    "max_norm": max_norm_based_estimation,
    "induction_head": induction_head_based_estimation,  # To be implemented
}


class LexicalCueEmbedder:
    def __init__(self, config: DictConfig, conflict_handler: ConflictConfigHandler):
        self.config = config
        self.conflict_handler = conflict_handler
        logger.info(f"Lexical cue estimation strategy: {self.config.est_strategy}")
        logger.info(f"Lexical cue filling strategy: {self.config.filling_strategy}")

    def estimate_lexical_tokens(self, kv_cache: EvictCache) -> torch.Tensor:
        """
        Estimate the positions of lexical cue tokens under the given estimation strategy.
        """
        est_strategy_func = EST_STRATEGY_FUNCS.get(self.config.est_strategy, None)
        if est_strategy_func is None:
            raise ValueError(f"Estimation strategy {self.config.est_strategy} is not supported.")
        kwargs = {}
        if self.config.est_strategy == "induction_head":
            kwargs["induction_head_indices"] = None
        lex_cue_positions = est_strategy_func(kv_cache, self.config.n_tokens, **kwargs)
        return lex_cue_positions

    def fill_guess_tokens(self, relevance: str, kv_cache: EvictCache, lex_cue_positions: torch.Tensor):
        pass

    def gate_lexical_cues(self, relevance: str, kv_cache: EvictCache, lex_cue_positions: torch.Tensor):
        target_layer_and_heads = self.conflict_handler.normal_map if relevance == "negative"\
                                else self.conflict_handler.critical_map
        # Selectively gate
        embed_value = True
        # embed_value = (relevance != "negative")
        pos_indices_t = lex_cue_positions.squeeze(0)
        for layer_idx, head_indices in target_layer_and_heads.items():
            head_indices_t = torch.tensor(head_indices, device=kv_cache.valid.device)
            kv_cache.valid[layer_idx, :, head_indices_t[:, None], pos_indices_t] = embed_value
        kv_cache.prepare_init() # Call prepare_init here (update pruned kv)
        return kv_cache

    def embed_lexical_cues(self, tagged_all_kv: List[Tuple[str, EvictCache]]) -> List[EvictCache]:
        kv_cache_list = []
        for tagged_kv_cache in tagged_all_kv:
            relevance, kv_cache = tagged_kv_cache
            lex_cue_positions = self.estimate_lexical_tokens(kv_cache)
            # kv_cache = self.fill_guess_tokens(relevance, kv_cache, lex_cue_positions)
            kv_cache = self.gate_lexical_cues(relevance, kv_cache, lex_cue_positions)
            kv_cache_list.append(kv_cache)
        return kv_cache_list