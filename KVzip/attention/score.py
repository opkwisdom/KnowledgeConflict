# ------------------------------------------------------------------------------
# Original Code developed by Jang-Hyun Kim
# Licensed under The MIT License
# GitHub Repository: https://github.com/snu-mllab/KVzip
# ------------------------------------------------------------------------------
import math
import torch
import torch.nn as nn
from typing import List, Tuple, Union, Optional, Dict


class KVScore():
    """ Functions to compute the score for the KV features. (kvcache.py)"""

    def __init__(self):
        self.n_heads_kv = None
        self.dtype = None
        self.device = None
        self.get_score = True
        self.causal_mask_score = None
        self.score = None
        self.raw_score = None
        self.sink = None
        self.start_idx, self.end_idx = None, None

    def init_score(self):
        self.get_score = True
        self.causal_mask_score = None
        self.score = [
            torch.zeros((1, self.n_heads_kv, 0), dtype=self.dtype, device=self.device)
            for _ in range(self.n_layers)
        ]
        self.raw_score = [
            torch.zeros((1, self.n_heads_kv, 0), dtype=self.dtype, device=self.device)
            for _ in range(self.n_layers)
        ]

    def _update_score(self, layer_idx: int, score: torch.Tensor, raw_score: torch.Tensor):
        self.score[layer_idx] = torch.cat([self.score[layer_idx], score], dim=-1)
        self.raw_score[layer_idx] = torch.cat([self.raw_score[layer_idx], raw_score], dim=-1)

    def _get_score(self, query_states: torch.Tensor, key_states: torch.Tensor, layer_idx: int):
        """ Compute KV importance scores.
            # key_states: bsz x head_kv x k x dim, query_states: bsz x head x q x dim
        """
        bsz, num_heads, q_len, head_dim = query_states.shape
        num_kv = key_states.size(1)

        query_states = query_states.view(bsz, num_kv, -1, q_len, head_dim)  # Split heads into groups
        key_states = torch.cat(
            [
                key_states[:, :, :self.sink],  # sink tokens (generally system prompt)
                key_states[:, :, self.start_idx:self.end_idx],  # KV chunk in the cache (Context)
                key_states[:, :, -q_len:],  # KV repeat chunk
            ],
            dim=2)

        # bsz, head, 1, dim, k
        # bsz, head_kv, n_c + n_in, dim
        key_states = key_states.unsqueeze(2).transpose(-2, -1).contiguous()
        ctx_len = self.end_idx - self.start_idx
        attn_weights = torch.matmul(query_states, key_states) / math.sqrt(head_dim)
        self._mask_causal(attn_weights, q_len)

        # bsz, head, group, q, ctx_len
        raw_score = attn_weights[..., self.sink:self.sink + ctx_len].amax(dim=(-3, -2))
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)  # not fp32
        attn_weights = attn_weights[..., self.sink:self.sink + ctx_len] # only consider the chunk in the cache
        score = attn_weights.amax(dim=(-3, -2))  # max over group, q

        self._update_score(layer_idx, score, raw_score)

    def _make_mask(self, attn_weights: torch.Tensor, window_size: int):
        """ Define causal mask shared across layers
        """
        mask = torch.full((window_size, window_size),
                          torch.finfo(attn_weights.dtype).min,
                          device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        self.causal_mask_score = mask[None, None, None, :, :]

    def _mask_causal(self, attn_weights: torch.Tensor, window_size: int):
        """ Apply causal maksing
        """
        if self.causal_mask_score is None:
            self._make_mask(attn_weights, window_size)
        elif self.causal_mask_score.size(-1) != window_size:
            self._make_mask(attn_weights, window_size)

        attn_weights[..., -window_size:, -window_size:] += self.causal_mask_score

    ##################################################################################################
    def _threshold(self, score: Union[torch.Tensor, List[torch.Tensor]], ratio: float, prune_kwargs: Optional[Dict[int, List[int]]] = None, prune_type: str = "positive"):
        """ 
        Apply thresholding to KV importance scores
        Prune Args can specify specific layers and heads to prune, e.g., ["5_1", "5_3", "6"]
        """
        # if type(score) == list:
        #     score = torch.stack(score, dim=0)
        
        # Parse prune_args
        L = len(score)
        _, H, N = score[0].shape  # layers, heads, seq_len

        target_layer_and_heads = {}
        if prune_kwargs is not None:
            target_layer_and_heads = prune_kwargs
        
        pruned_score, thres_list = [], []
        for layer_idx, layer_score in enumerate(score):
            if layer_idx in target_layer_and_heads:
                # Head-wise thresholding
                target_heads = target_layer_and_heads[layer_idx]
                target_score = layer_score[:, target_heads].reshape(-1)

                if ratio < 1:
                    score_sort = torch.sort(target_score, descending=True).values
                    n = max(int(len(score_sort) * ratio) - 1, 0)
                    thres = score_sort[n].item()

                    # Default: keep for positive, remove for negative
                    valids = torch.ones_like(layer_score, dtype=torch.bool)
                    if prune_type == "positive":
                        target_valids = (layer_score[:, target_heads] > thres).bool()
                    else:
                        target_valids = (layer_score[:, target_heads] < thres).bool()
                    valids[:, target_heads] = target_valids
                else:
                    valids = torch.ones_like(layer_score, dtype=bool)
                pruned_score.append(valids)
                thres_list.append(thres)
            else:
                pruned_score.append(torch.ones_like(layer_score, dtype=bool))
                thres_list.append(0)
        
        pruned_kv = torch.stack(pruned_score, dim=0)    # (L, 1, H, N)
        # if ratio < 1:
        #     score_sort = torch.sort(score.reshape(-1), descending=True).values
        #     n = max(int(len(score_sort) * ratio) - 1, 0)
        #     thres = score_sort[n].item()
        #     valids = torch.where(score > thres, True, False).bool()
        # else:
        #     valids = torch.ones_like(score, dtype=bool)
        #     thres = 0.

        return pruned_kv, thres_list

    def _threshold_uniform(self, scores: Union[torch.Tensor, List[torch.Tensor]], ratio: float):
        """ Apply thresholding to KV importance scores with uniform head budgets 
        """
        valids = []
        for nl, score in enumerate(scores):
            if ratio < 1:
                n_seq = score.size(-1)
                k = int(n_seq * ratio)
                _, topk_indices = torch.topk(score, k, dim=-1)
                valid = torch.zeros_like(score, dtype=bool)
                valid.scatter_(-1, topk_indices, True)
            else:
                valid = torch.ones_like(score, dtype=bool)
            valids.append(valid)

        valids = torch.stack(valids)
        return valids, 0
    
    def validate_relevance(
        self,
        topk: int = 100,
        control_cache_stats: Dict[int, Dict[str, float]] = None,
        control_d: List[float] = None,
    ) -> Tuple[bool, float]:
        """
        Method to analyze the computed scores
        """
        # Extract top-k diff scores
        sample_topk_diff_scores = torch.zeros((self.n_layers, topk), dtype=torch.float32)

        for i, layer_score in enumerate(self.raw_score):
            score_flat = layer_score.reshape(-1)
            topk_scores = torch.topk(score_flat, topk, largest=True).values
            mean_score = torch.mean(score_flat)
            topk_diff_scores = topk_scores - mean_score
            sample_topk_diff_scores[i] = topk_diff_scores

        # Compute conflict scores (Cohen's d)
        d_layers = [int(layer) for layer in control_d.keys()]
        x_mu = sample_topk_diff_scores.mean(dim=1)
        x_std = sample_topk_diff_scores.std(dim=1)
        
        accept_ratio = 0
        for layer_idx in d_layers:
            x = {
                "mean": x_mu[layer_idx].item(),
                "std": x_std[layer_idx].item(),
            }
            y = control_cache_stats[layer_idx]
            d = cohens_d(x, y)
            if d >= control_d[layer_idx]:
                accept_ratio += 1
        accept_ratio /= len(d_layers)

        if accept_ratio >= 0.5:
            return True, accept_ratio
        else:
            return False, accept_ratio

class HybridKVScore(KVScore):

    def init_score(self):
        self.get_score = True
        self.causal_mask_score = None

        self.score = [
            torch.zeros((1, self.n_heads_kv, 0), dtype=self.dtype, device=self.device)
            for _ in range(self.num_static_layers)
        ]
    
    def _get_score(self, query_states, key_states, layer_idx):
        if layer_idx in self.layer_id_to_static_id:
            static_layer_idx = self.layer_id_to_static_id[layer_idx]
            super()._get_score(query_states, key_states, static_layer_idx)

def cohens_d(x, y):
    """
    Compute Cohen's d between two distributions
    """
    pooled_std = ((x["std"] ** 2 + y["std"] ** 2) / 2) ** 0.5
    d = abs(x["mean"] - y["mean"]) / pooled_std
    return d