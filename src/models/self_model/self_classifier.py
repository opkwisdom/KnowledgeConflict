import torch
import torch.nn as nn
from omegaconf import DictConfig


### Self Simple Classifier for GenLoss Prediction (for ablation) ###
class SelfGGClassifier(nn.Module):
    def __init__(self, hidden_size=4096, dropout_prob=0.1, pooling_strategy="mean"):
        super().__init__()
        self.pooling_strategy = pooling_strategy
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, 1)
        )

    def pooling(self, query_hidden_states: torch.FloatTensor):
        if self.pooling_strategy == "mean":
            # mean pooling
            pooled_output = query_hidden_states.mean(dim=1)
        elif self.pooling_strategy == "max":
            # most reactive token
            pooled_output = query_hidden_states.max(dim=1).values
        else:
            raise NotImplementedError(f"Pooling strategy {self.pooling_strategy} not implemented.")
        return pooled_output

    def forward(self, query_hidden_states: torch.FloatTensor):
        pooled_output = self.pooling(query_hidden_states)   # (B, D_llm)
        logits = self.classifier(pooled_output)  # (B, 1)
        return logits.squeeze(-1)  # (B,)



class SelfGenLossModel(nn.Module):
    def __init__(self, llm, n_soft_tokens=32, hidden_dim=4096):
        super().__init__()
        self.llm = llm  # frozen
        self.n_soft_tokens = n_soft_tokens
        self.soft_tokens = nn.Parameter(torch.randn(n_soft_tokens, hidden_dim) * 0.02)
        self.classifier = SelfGGClassifier(hidden_dim, dropout_prob=0.1, pooling_strategy="mean")
    
    def forward(self, input_ids, attention_mask):
        # input_ids: (B*N, S) - [Context_i + Query]
        B_N = input_ids.size(0)
        
        # Get input embeddings
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)  # (B*N, S, D)
        
        # Append soft tokens
        soft = self.soft_tokens.unsqueeze(0).expand(B_N, -1, -1)  # (B*N, M, D)
        inputs_embeds = torch.cat([inputs_embeds, soft], dim=1)  # (B*N, S+M, D)
        
        # Extend attention mask
        soft_mask = torch.ones(B_N, self.n_soft_tokens, device=attention_mask.device)
        attention_mask = torch.cat([attention_mask, soft_mask], dim=1)
        
        # Forward through LLM
        outputs = self.llm.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True
        )
        
        # Take last M positions (soft token outputs)
        soft_hidden = outputs.last_hidden_state[:, -self.n_soft_tokens:, :]  # (B*N, M, D)
        score = self.classifier(soft_hidden)  # (B*N, 1)
        return score