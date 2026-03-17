import torch
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):
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