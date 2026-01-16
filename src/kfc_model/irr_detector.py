import torch
import torch.nn as nn

class SimpleIrrDetector(nn.Module):
    def __init__(self, num_layers: int, top_k: int):
        super().__init__()
        self.input_dim = num_layers * top_k
        self.register_buffer('running_mean', torch.zeros(self.input_dim))
        self.register_buffer('running_std', torch.ones(self.input_dim))
        self.fitted = False

        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim // 2),
            nn.BatchNorm1d(self.input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.input_dim // 2, self.input_dim // 8),
            nn.ReLU(),
            nn.Linear(self.input_dim // 8, 1)
        )

    def fit_scaler(self, X_train: torch.Tensor):
        """
        X_train: (N, Layers, TopK)
        """
        # Flatten
        flat_data = X_train.view(X_train.size(0), -1)
        
        mean = flat_data.mean(dim=0)
        std = flat_data.std(dim=0)
        std[std == 0] = 1.0
        
        self.running_mean.copy_(mean)
        self.running_std.copy_(std)
        self.fitted = True
    
    def forward(self, kv_score: torch.Tensor) -> torch.Tensor:
        """
        Args:
            kv_score: (batch_size, num_layers, top_k)
        Returns:
            irr_probs: (batch_size,)
        """
        batch_size = kv_score.shape[0]
        agg_scores = kv_score.view(batch_size, -1)  # (batch_size, num_layers * top_k)
        
        # Classify irregularity
        agg_scaled = (agg_scores - self.running_mean) / (self.running_std + 1e-6)
        return self.classifier(agg_scaled).squeeze(-1)  # (batch_size,)