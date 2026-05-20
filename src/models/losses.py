import torch
import einops
import torch.nn as nn
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
    
    
class RankwiseGuideLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input_scores: torch.FloatTensor, target_scores: torch.FloatTensor):
        """
        Args:
            input: Tensor of shape (B, K)
            target: Tensor of shape (B, K)
        Returns:
            loss: Scalar tensor representing the rankwise score loss
        """
        B, K = input_scores.shape
        input_diff = input_scores.unsqueeze(2) - input_scores.unsqueeze(1)  # (B, K, K)
        target_diff = target_scores.unsqueeze(2) - target_scores.unsqueeze(1)  # (B, K, K)
        right_term = torch.abs(target_diff) - torch.sign(target_diff) * input_diff
        rank_matrix = F.relu(right_term)
        denominator = K * (K - 1) if K > 1 else 1
        batch_loss = rank_matrix.sum(dim=(1, 2)) / denominator
        loss = batch_loss.mean()
        return loss
    

class ListwiseGuideLoss(torch.nn.Module):
    def __init__(self, T: float):
        super().__init__()
        self.T = T
    
    def forward(self, input_scores: torch.FloatTensor, target_scores: torch.FloatTensor):
        """
        Args:
            input: Tensor of shape (B, K)
            target: Tensor of shape (B, K)
        Returns:
            loss: Scalar tensor representing the rankwise score loss
        """
        input_scores = input_scores.float()
        target_scores = target_scores.float()

        target_log_prob = torch.log_softmax(target_scores / self.T, dim=-1).detach()  # (B, K)
        target_prob = target_log_prob.exp()

        input_log_prob = torch.log_softmax(input_scores / self.T, dim=-1)  # (B, K)
        input_prob = input_log_prob.exp()
        # loss = -(self.T ** 2) * (target_prob * input_log_prob).sum(dim=-1).mean()   # Forward KL
        loss = (self.T ** 2) * (input_prob * (input_log_prob - target_log_prob)).sum(dim=-1).mean()  # Reverse KL
        return loss


class PairwiseRankGuideLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, input_scores: torch.FloatTensor, target_scores: torch.FloatTensor):
        """
        Args:
            input: Tensor of shape (B, N)
            target: Tensor of shape (B, N)
        Returns:
            loss: Scalar tensor representing the rankwise score loss
        """
        input_diff = (input_scores.unsqueeze(2) - input_scores.unsqueeze(1)) * 10  # (B, N, N)
        target_diff = target_scores.unsqueeze(2) - target_scores.unsqueeze(1)  # (B, N, N)
        
        target_labels = (target_diff > 0).float()
        valid_mask = (target_diff != 0).float()
        raw_loss = self.loss_fn(input_diff, target_labels)
        masked_loss = raw_loss * valid_mask
        loss = masked_loss.sum() / valid_mask.sum().clamp(min=1e-6)
        return loss
    

class LambdaLoss(torch.nn.Module):
    def __init__(self, sigma: float = 1.0, eps: float = 1e-10, reduction: str = 'mean', k: int = 10):
        super().__init__()
        self.sigma = sigma
        self.eps = eps
        self.reduction = reduction
        self.k = k

    def forward(self, input_scores: torch.FloatTensor, target_scores: torch.FloatTensor):
        """
        Args:
            input: Tensor of shape (B, N)
            target: Tensor of shape (B, N)
        Returns:
            loss: Scalar tensor representing the rankwise score loss
        """
        input_scores = input_scores.float()
        target_scores = target_scores.float()
        
        B, N = input_scores.shape
        device = input_scores.device
        # ---------- Gain from teacher score ----------
        # Use ReLU to convert continuous score to non-negative gain and per-query normalize
        gains = torch.relu(target_scores)  # (B, N)
        gains = gains / gains.max(dim=-1, keepdim=True).values.clamp(min=self.eps)
        
        # ---------- Ideal DCG (for normalization) ----------
        ideal_gains, _ = gains.sort(dim=-1, descending=True)  # (B, N)
        positions = torch.arange(1, N + 1, device=device, dtype=torch.float)
        ideal_discounts = 1.0 / torch.log2(positions + 1.0)  # (N,)
        
        # Apply k truncation if specified
        if self.k is not None and self.k < N:
            ideal_dcg = (ideal_gains[:, :self.k] * ideal_discounts[:self.k]).sum(dim=-1, keepdim=True)
        else:
            ideal_dcg = (ideal_gains * ideal_discounts).sum(dim=-1, keepdim=True)
        ideal_dcg = ideal_dcg + self.eps  # (B, 1)
        
        # ---------- Predicted ranks (1-indexed) ----------
        pred_ranks = input_scores.argsort(dim=-1, descending=True).argsort(dim=-1).float() + 1.0  # (B, N)
        pred_discounts = 1.0 / torch.log2(pred_ranks + 1.0)  # (B, N)
        
        # ---------- Pairwise tensors ----------
        score_diff = input_scores.unsqueeze(-1) - input_scores.unsqueeze(-2)  # (B, N, N)
        target_diff = gains.unsqueeze(-1) - gains.unsqueeze(-2)  # (B, N, N)
        pair_mask = (target_diff > 0).float()  # (B, N, N)
        
        # ---------- NDCG delta (lambda weight) ----------
        gain_diff = torch.abs(gains.unsqueeze(-1) - gains.unsqueeze(-2))  # (B, N, N)
        discount_diff = torch.abs(pred_discounts.unsqueeze(-1) - pred_discounts.unsqueeze(-2))  # (B, N, N)
        delta_ndcg = (gain_diff * discount_diff) / ideal_dcg.unsqueeze(-1)  # (B, N, N)
        
        # ---------- k truncation for lambda weights ----------
        if self.k is not None and self.k < N:
            in_topk = (pred_ranks <= self.k).float()  # (B, K)
            topk_mask = in_topk.unsqueeze(-1) + in_topk.unsqueeze(-2)  # (B, K, K), at least one in top-k
            topk_mask = (topk_mask > 0).float()
            delta_ndcg = delta_ndcg * topk_mask
        
        # ---------- Pairwise loss ----------
        pair_loss = torch.nn.functional.softplus(-self.sigma * score_diff)  # (B, N, N)
        weighted_loss = delta_ndcg * pair_loss * pair_mask  # (B, N, N)
        loss_per_query = weighted_loss.sum(dim=(-1, -2))  # (B,)
        n_pairs = pair_mask.sum(dim=(-1, -2)).clamp(min=1.0)
        loss_per_query = loss_per_query / n_pairs
        
        if self.reduction == "mean":
            return loss_per_query.mean()
        elif self.reduction == "sum":
            return loss_per_query.sum()
        else:
            return loss_per_query