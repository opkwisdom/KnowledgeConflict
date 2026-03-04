import torch
import torch.nn as nn
from typing import Union
from omegaconf import DictConfig

from .modeling_ca_former import SingleHiddenCAFormer, MultiHiddenCAFormer


class CAFormerClassifier(nn.Module):
    def __init__(self, config: DictConfig, ca_former: Union[SingleHiddenCAFormer, MultiHiddenCAFormer]):
        super().__init__()
        self.config = config
        self.ca_former = ca_former

        dropout_prob = getattr(self.config.caformer, "dropout", 0.1)
        self.classifier = self.classifier = nn.Sequential(
            nn.Linear(self.config.caformer.llm_width, self.config.caformer.llm_width),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(self.config.caformer.llm_width, 3)
        )

    def pooling(self, query_hidden_states: torch.FloatTensor):
        if self.config.caformer.pooling_strategy == "mean":
            # mean pooling
            pooled_output = query_hidden_states.mean(dim=1)
        elif self.config.caformer.pooling_strategy == "max":
            # most reactive token
            pooled_output = query_hidden_states.max(dim=1).values
        else:
            raise NotImplementedError(f"Pooling strategy {self.config.caformer.pooling_strategy} not implemented.")
        return pooled_output

    def forward(self, llm_hidden_states: torch.FloatTensor, attention_mask: torch.LongTensor, output_hidden_states: bool = True):
        """
        Args:
            llm_hidden_states: Tensor of shape (B, L_select, S, D_llm) or (B, S, D_llm)
            attention_mask: Tensor of shape (B, K)
        Returns:
            logits: Tensor of shape (B, 3)
        """
        query_hidden_states = self.ca_former(llm_hidden_states, attention_mask)    # (B, K, D_llm)
        pooled_output = self.pooling(query_hidden_states)   # (B, D_llm)

        logits = self.classifier(pooled_output)  # (B, 3)
        outputs = (logits,)
        if output_hidden_states:
            outputs += (pooled_output,)
        return outputs
