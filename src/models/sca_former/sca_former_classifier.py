import torch
import torch.nn as nn
from typing import Union
from omegaconf import DictConfig

from .modeling_sca_former import SingleHiddenSCAFormer, MultiHiddenSCAFormer


class SCAFormerClassifier(nn.Module):
    def __init__(self, config: DictConfig, sca_former: Union[SingleHiddenSCAFormer, MultiHiddenSCAFormer]):
        super().__init__()
        self.config = config
        self.sca_former = sca_former

        self.classifier = nn.Linear(config.kvformer.llm_width, 3)

    def forward(self, kvformer_input: torch.FloatTensor, kvformer_mask: torch.LongTensor):
        """
        Args:
            kvformer_input: Tensor of shape (B, L_select, S, D)
            kvformer_mask: Tensor of shape (B, S)
        Returns:
            logits: Tensor of shape (B, 3)
        """
        query_hidden_states = self.sca_former(kvformer_input, kvformer_mask)    # (B, K, D)