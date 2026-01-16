import torch
from omegaconf import DictConfig
from typing import List, Tuple

from KVzip.attention import EvictCache

class LexicalKeeper:
    def __init__(self, config: DictConfig):
        self.config = config

    def estimate_lexical_tokens(self, kv_cache: EvictCache):
        pass

    def fill_guess_tokens(self, kv_cache: EvictCache, lex_token_positions: torch.Tensor):
        pass