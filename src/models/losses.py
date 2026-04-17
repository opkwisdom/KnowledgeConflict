import torch
import einops
import torch.nn.functional as F

class SCIContrastiveLoss(torch.nn.Module):
    """
    Custom SupCon loss for S/C/I labeling dataset.
    """
    def __init__(self, T: float = 1.0):
        super().__init__()
        self.T = T
    
    def forward(self, input: torch.FloatTensor, target: torch.LongTensor):
        """
        Args:
            input: Tensor of shape (B, D)
            target: Tensor of shape (B,), where each value is in {0, 1, 2}
                    representing positive, negative, irrelevant
        Returns:
            loss: Scalar tensor representing the contrastive loss
        """
        input = F.normalize(input, p=2, dim=1)
        batch_scores = torch.exp(torch.matmul(input, input.T) / self.T) # (B, B)
        base_mask = torch.ones_like(batch_scores) \
            - torch.eye(batch_scores.size(0), device=batch_scores.device) # exclude self-similarity

        group_mask = (target.unsqueeze(1) == target.unsqueeze(0))   # S/C/I group mask
        valid_positive_mask = (target != 2).unsqueeze(1)
        group_mask = group_mask & valid_positive_mask   # Only consider positives excluding irrelevant samples
        batch_mask = base_mask * group_mask.float()

        denom = torch.sum(batch_scores * base_mask, dim=1, keepdim=True)    # (B, 1)
        log_prob = torch.log(batch_scores / (denom + 1e-9))
        num_positives = torch.sum(batch_mask, dim=1)   # (B,)

        loss = -torch.sum(log_prob * batch_mask, dim=1) / (num_positives + 1e-9)   # (B,)
        loss = loss.mean()
        return loss
    


class MultiQueryContrastiveLoss(torch.nn.Module):
    def __init__(self, T: float = 1.0):
        super().__init__()
        self.T = T
    
    def forward(self, input: torch.FloatTensor, target: torch.FloatTensor):
        """
        Args:
            input: Tensor of shape (B, K, D)
            target: Tensor of shape (B, D)
        Returns:
            max_loss: Scalar tensor representing the max contrastive loss across K queries
            mean_loss: Scalar tensor representing the mean contrastive loss across K queries
        """
        B, K, D = input.shape
        # L2 normalization
        input = F.normalize(input, p=2, dim=-1)
        target = F.normalize(target, p=2, dim=-1)

        # sim_matrix: (B, B, K)
        sim_matrix = einops.einsum(input, target, "b1 k d, b2 d -> b1 b2 k") / self.T
        max_logits = sim_matrix.max(dim=-1).values  # (B, B)
        mean_logits = sim_matrix.mean(dim=-1)       # (B, B)

        labels = torch.arange(B, device=input.device)
        max_loss = F.cross_entropy(max_logits, labels)
        mean_loss = F.cross_entropy(mean_logits, labels)

        preds = max_logits.argmax(dim=1)
        acc = (preds == labels).float().mean().item()
        
        return max_loss, mean_loss, acc


class CosSimRegLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input: torch.FloatTensor):
        """
        Args:
            input: Tensor of shape (B, K, D)
        Returns:
            reg_loss: Scalar tensor representing the cosine similarity regularization loss
        """
        B, K, D = input.shape
        input = F.normalize(input, p=2, dim=-1)

        sim_matrix = torch.bmm(input, input.transpose(1, 2))    # (B, K, K)
        mask = torch.eye(K, dtype=torch.bool, device=input.device).unsqueeze(0)  # (1, K, K)
        sim_matrix = sim_matrix.masked_fill(mask, 0.0)
        mean_sim = sim_matrix.sum() / (B * K * (K - 1))
        return mean_sim