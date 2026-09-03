"""Loss function: MultiTaskBayesianRankingLoss."""

from typing import List, Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NUM_TASKS, TASK_NAMES


class MultiTaskBayesianRankingLoss(nn.Module):
    """
    For each task t ∈ {mu_e, mu_h} compute:
      1. Bayesian Personalized Ranking (BPR) Loss:
           L_bpr = -log( σ( sign(y1_t - y2_t) * (s1_t - s2_t) ) )
         where σ is the sigmoid function. Maximizes log-likelihood of correct ranking.
      2. Delta Regression Loss:
           L_reg  = MSE( s1_t - s2_t,  y1_t - y2_t )

    Total loss = Σ_t λ_t * (α · L_bpr_t + β · L_reg_t)
    """

    def __init__(
        self,
        rank_weight: float = 0.6,
        reg_weight: float = 0.4,
        task_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.alpha = rank_weight
        self.beta = reg_weight
        self.task_w = task_weights or [1.0] * NUM_TASKS

    def forward(
        self,
        s1: torch.Tensor,
        s2: torch.Tensor,
        y1: torch.Tensor,
        y2: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        log: Dict[str, float] = {}
        task_losses = []

        for t, (name, lam) in enumerate(zip(TASK_NAMES, self.task_w)):
            diff_s = s1[:, t] - s2[:, t]
            diff_y = y1[:, t] - y2[:, t]

            sign = diff_y.sign()
            nonzero_mask = sign != 0
            if nonzero_mask.any():
                sign_nonzero = sign[nonzero_mask]
                diff_s_nonzero = diff_s[nonzero_mask]
                l_bpr = F.softplus(-sign_nonzero * diff_s_nonzero).mean()
            else:
                l_bpr = (diff_s * 0.0).sum()

            l_reg = F.mse_loss(diff_s, diff_y)

            task_loss = lam * (self.alpha * l_bpr + self.beta * l_reg)
            task_losses.append(task_loss)

            log[f"{name}_bpr"] = l_bpr.item()
            log[f"{name}_reg"] = l_reg.item()

        total = sum(task_losses)
        log["total"] = total.item()
        return total, log
