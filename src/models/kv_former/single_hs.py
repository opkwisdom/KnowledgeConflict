import torch
import torch.nn as nn
from omegaconf import DictConfig
from transformers import AutoTokenizer, AutoConfig

from KVformer_roberta import RobertaModel
from KVformer_config import KVFormerConfig


class SingleHiddenKVFormer(nn.Module):
    def __init__(self, cfg: DictConfig, llm_tokenizer: AutoTokenizer):
        super().__init__()

        self.cfg = cfg
        self.query_length = cfg.query_length
        self.llm_tokenizer = llm_tokenizer

        self.roberta_config = KVFormerConfig.from_pretrained(
            cfg.model_name_or_path,
            query_length=cfg.query_length,
            llm_width=cfg.llm_width
        )
        self.model = RobertaModel.from_pretrained(
            cfg.model_name_or_path,
            config=self.roberta_config,
        )
        self.model.requires_grad_(True)

        self.query_token_embeds = nn.Parameter(torch.zeros(self.num_query_tokens, self.roberta_config.hidden_size))
        self.query_token_embeds.data.normal_(mean=0.0, std=self.roberta_config.initializer_range)
    
    def gen_query_embeds(self, batch_size: int):
        query_embeds = self.query_token_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        return query_embeds
    
    def forward(self, llm_hidden_states: torch.FloatTensor):
        x=1
        pass


# if __name__ == "__main__":
#     llm_model_name_or_path = "meta-llama/Llama-3.1-8B-Instruct"
#     llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name_or_path)

#     cfg = DictConfig({
#         "model_name_or_path": "roberta-base",
#         "query_length": 8,
#         "llm_width": 4096,
#     })
#     model = SingleHiddenKVFormer(cfg, llm_tokenizer)
