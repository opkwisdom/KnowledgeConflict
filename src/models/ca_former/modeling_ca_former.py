import torch
import torch.nn as nn
from omegaconf import DictConfig
from transformers import AutoTokenizer, AutoConfig

from .ca_former_roberta import RobertaModel
from .ca_former_config import CAFormerConfig


class SingleHiddenCAFormer(nn.Module):
    def __init__(self, cfg: DictConfig, llm_tokenizer: AutoTokenizer):
        super().__init__()

        self.cfg = cfg
        self.query_length = cfg.query_length
        self.llm_tokenizer = llm_tokenizer

        self.roberta_config = CAFormerConfig.from_pretrained(
            cfg.model_name_or_path,
            query_length=cfg.query_length,
            llm_width=cfg.llm_width
        )
        self.model = RobertaModel.from_pretrained(
            cfg.model_name_or_path,
            config=self.roberta_config,
            torch_dtype=torch.bfloat16,
            add_pooling_layer=False,
        )
        self.model.requires_grad_(True)

        self.query_token_embeds = nn.Parameter(torch.zeros(self.query_length, self.roberta_config.hidden_size))
        self.query_token_embeds.data.normal_(mean=0.0, std=self.roberta_config.initializer_range)

        self.linear_proj = nn.Linear(
            self.roberta_config.hidden_size,
            self.roberta_config.llm_width
        )
    
    def gen_query_embeds(self, llm_hidden_states: torch.FloatTensor):
        batch_size = llm_hidden_states.size(0)
        query_embeds = self.query_token_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        return query_embeds
    
    def forward(self, llm_hidden_states: torch.FloatTensor, attention_mask: torch.LongTensor):
        """
        Args:
            llm_hidden_states: Tensor of shape (B, L_select, S, D_llm), use only single layer
            attention_mask: Tensor of shape (B, S)
        Returns:
            query_hidden_states: Tensor of shape (B, K, D_llm)
        """
        llm_hidden_states = llm_hidden_states[:, 0]     # single layer
        query_embeds = self.gen_query_embeds(llm_hidden_states)

        llm_hidden_states = llm_hidden_states.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)

        outputs = self.model(
            query_embeds=query_embeds,
            encoder_hidden_states=llm_hidden_states,
            encoder_attention_mask=attention_mask,
        )
        sequence_output = outputs.last_hidden_state # (B, K, D_probe)
        query_hidden_states = self.linear_proj(sequence_output)
        return query_hidden_states


class MultiHiddenCAFormer(nn.Module):
    def __init__(self, cfg: DictConfig, llm_tokenizer: AutoTokenizer):
        super().__init__()

# if __name__ == "__main__":
#     llm_model_name_or_path = "meta-llama/Llama-3.1-8B-Instruct"
#     llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name_or_path)

#     cfg = DictConfig({
#         "model_name_or_path": "roberta-base",
#         "query_length": 8,
#         "llm_width": 4096,
#     })
#     model = SingleHiddenKVFormer(cfg, llm_tokenizer)
