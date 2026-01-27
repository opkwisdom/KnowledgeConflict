import torch

from typing import Tuple
from src.utils import BoostedProbResult

class UncertaintyEstimator:
    def __init__(self, config):
        self.config = config

    def calibrate(self, scores: Tuple[torch.Tensor]) -> float:
        """
        Return calibrated probs for inference-time estimation.
        """
        scores = torch.cat(scores, dim=0)   # (seq_len, vocab_size)
        probs = torch.softmax(scores, dim=-1)

        x = self.config.x
        eps = self.config.eps

        sorted_prob, indices = torch.sort(probs, dim=-1, descending=True)
        padded = torch.zeros((sorted_prob.shape[0], 1), device=sorted_prob.device)
        probs_diff = sorted_prob - torch.cat([sorted_prob[:, 1:], padded], dim=1)

        # calibration logic
        is_significant_drop = (probs_diff >= sorted_prob * x) & (probs_diff >= eps)
        vocab_size = sorted_prob.shape[1]
        total_indices = torch.arange(vocab_size, device=sorted_prob.device).unsqueeze(0)
        max_true_indices = (total_indices * is_significant_drop.int()).max(dim=1).values

        probs_cumsum = torch.cumsum(sorted_prob, dim=-1)
        dominant_cluster = torch.gather(probs_cumsum, dim=1, index=max_true_indices.unsqueeze(-1)).squeeze(-1)

        return dominant_cluster.mean().item()

    def calibrate_inspect(self, scores: Tuple[torch.Tensor]) -> BoostedProbResult:
        """
        Inspect calibrated probs for hyperparameter tuning.
        """
        scores = torch.cat(scores, dim=0)   # (seq_len, vocab_size)
        probs = torch.softmax(scores, dim=-1)

        top_k = self.config.top_k
        top_p = self.config.top_p
        if top_k and top_p:
            raise ValueError("Only one of top_k and top_p should be set.")
        
        sorted_prob, indices = torch.sort(probs, dim=-1, descending=True)
        padded = torch.zeros((sorted_prob.shape[0], 1), device=sorted_prob.device)
        probs_diff = sorted_prob - torch.cat([sorted_prob[:, 1:], padded], dim=1)

        if top_k:
            diff_probs = probs_diff[:, :top_k]
            accum_prob = diff_probs.sum(dim=-1)
        else:
            cum_prob = torch.cumsum(sorted_prob, dim=-1)
            mask = cum_prob <= top_p
            mask_with_first_p = mask.clone()
            first_false_idx = (~mask).int().argmax(dim=-1)
            mask_with_first_p.scatter_(1, first_false_idx.unsqueeze(-1), True)
            accum_prob = (sorted_prob * mask_with_first_p).sum(dim=-1)
        
        return BoostedProbResult(
            top_k=top_k if top_k is not None else -1,
            top_p=top_p if top_p is not None else -1.0,
            accum_prob=accum_prob.cpu(),
            diff_probs=diff_probs.cpu()
        )



# probs_diff = sorted_prob - torch.cat([sorted_prob[:, :1], torch.zeros()], dim=1)