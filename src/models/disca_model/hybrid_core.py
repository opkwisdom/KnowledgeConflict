from omegaconf import DictConfig
from typing import Any, Dict, List
import torch
import logging
from sentence_transformers import CrossEncoder

from .core import DISCA
from utils import CtxExample
from ..ca_former import CAFormerGGClassifier


logger = logging.getLogger(__name__)


class HybridDISCA(DISCA):
    """
    Distillation-based Integrated Scoring via CA-Former
    """
    def __init__(self, config: DictConfig, model_name: str, caformer_clf: CAFormerGGClassifier, ce_reranker: CrossEncoder) -> None:
        super().__init__(config, model_name, caformer_clf)
        self.ce_reranker = ce_reranker
        # ["rrf", "concat", "interleave"]
        self.fusion_method = getattr(config, "fusion_method", "rrf")
        self.fusion_weight = getattr(config, "fusion_weight", 0.5)
        self.rrf_k = getattr(config, "rrf_k", 60)
    
    def _compute_sample_ce_scores(
        self,
        query: str,
        ctxs: List[CtxExample]
    ) -> torch.Tensor:
        # Prepare pairs for reranking
        passages = [f"Title: {ctx.title}\n\n{ctx.text}" for ctx in ctxs]
        scores = self.ce_reranker.predict([(query, passage) for passage in passages],
                                        convert_to_numpy=False, convert_to_tensor=True, show_progress_bar=False)
        return scores

    def _compute_ce_scores(
        self,
        queries: List[str],
        contexts_list: List[List[CtxExample]],
    ) -> torch.Tensor:
        # Prepare pairs for reranking
        batch_scores = []
        for query, ctxs in zip(queries, contexts_list):
            sample_scores = self._compute_sample_ce_scores(query, ctxs)
            batch_scores.append(sample_scores)
        return torch.stack(batch_scores).to(self.ce_reranker.device)     # (B, N)

    def compute_rerank_scores(
        self,
        inputs: Dict[str, Any],
        queries: List[str],
        contexts_list: List[List[CtxExample]],
        mg_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute final rerank scores
        Fuse MG scores with CE scores
        """
        ce_scores = self._compute_ce_scores(queries, contexts_list)
        B, N = ce_scores.shape
        mg_scores = mg_scores.reshape(B, N)
        
        # Fuse MG scores with CE scores
        assert mg_scores.shape == ce_scores.shape, \
            "MG scores and CE scores must have the same shape"
        if self.fusion_method == "rrf":
            return self._rrf_fuse(mg_scores, ce_scores)
        elif self.fusion_method == "concat":
            return self._concat_fuse(mg_scores, ce_scores)
        elif self.fusion_method == "interleave":
            return self._interleave_fuse(mg_scores, ce_scores)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

    ##### Fusion Methods #####
    def _convert_scores_to_rank(self, scores):
        """
        1-indexed ranks along the last dimension.
        """
        return scores.argsort(dim=-1, descending=True).argsort(dim=-1) + 1

    def _rrf_fuse(self, mg_scores, ce_scores):
        mg_ranks = self._convert_scores_to_rank(mg_scores).float()
        ce_ranks = self._convert_scores_to_rank(ce_scores).float()
        
        mg_rr = 1 / (self.rrf_k + mg_ranks)
        ce_rr = 1 / (self.rrf_k + ce_ranks)
        final_scores = mg_rr + ce_rr
        return final_scores