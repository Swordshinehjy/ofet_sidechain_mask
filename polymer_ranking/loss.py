"""Loss function: MultiTaskBayesianRankingLoss (with censored-label support)."""

from typing import List, Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NUM_TASKS, TASK_NAMES


class MultiTaskBayesianRankingLoss(nn.Module):
    """Ranking + delta-regression loss for two mobility tasks.

    Label semantics
    ---------------
    A mobility of ``0`` in this dataset means *below the detection limit*
    (left-censored), not "not measured". Therefore:

    * the **ordering is still decidable**: a measured value is always above the
      detection limit, so ``measured > censored`` is a valid ranking constraint;
    * the **numeric difference is unknown**, so such pairs must not supervise the
      delta regression term.

    Consequently two masks are used:

    ``valid``
        both polymers were measured (``> 0``) → delta regression.
    ``rank_valid``
        the ordering is decidable → BPR. This includes pairs with one censored
        side, which are weighted by ``censored_weight`` (they carry direction but
        no magnitude).

    Loss per task: ``lambda_t * (alpha * L_bpr_t + beta * L_reg_t)`` with

    .. math::
        L_{bpr} = -\\log \\sigma(\\text{sign}(y_1-y_2)(s_1-s_2)), \\qquad
        L_{reg} = \\text{MSE}(s_1-s_2,\\, (y_1-y_2)/\\text{delta\\_scale})
    """

    def __init__(
        self,
        rank_weight: float = 0.8,
        reg_weight: float = 0.2,
        task_weights: Optional[List[float]] = None,
        delta_scale: float = 1.0,
        censored_weight: float = 0.5,
    ):
        super().__init__()
        self.alpha = rank_weight
        self.beta = reg_weight
        self.task_w = task_weights or [1.0] * NUM_TASKS
        self.delta_scale = float(delta_scale) if delta_scale else 1.0
        self.censored_weight = float(censored_weight)

    def forward(
        self,
        s1: torch.Tensor,
        s2: torch.Tensor,
        y1: torch.Tensor,
        y2: torch.Tensor,
        valid: Optional[torch.Tensor] = None,
        rank_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        log: Dict[str, float] = {}
        task_losses = []

        for t, (name, lam) in enumerate(zip(TASK_NAMES, self.task_w)):
            diff_s = s1[:, t] - s2[:, t]
            diff_y = y1[:, t] - y2[:, t]

            if valid is None:
                v_reg = torch.ones_like(diff_s, dtype=torch.bool)
                v_rank = diff_y != 0
            else:
                v_reg = valid[:, t].to(torch.bool)
                src = rank_valid if rank_valid is not None else valid
                v_rank = src[:, t].to(torch.bool) & (diff_y != 0)

            # ---- delta regression: only pairs with two measured values ----
            if int(v_reg.sum()) == 0:
                l_reg = 0.0 * diff_s.sum()
            else:
                l_reg = F.mse_loss(diff_s[v_reg], diff_y[v_reg] / self.delta_scale)

            # ---- BPR: every pair with a decidable ordering ----
            if int(v_rank.sum()) == 0:
                l_bpr = 0.0 * diff_s.sum()
            else:
                ds = diff_s[v_rank]
                dy = diff_y[v_rank]
                w = torch.ones_like(ds)
                if self.censored_weight != 1.0 and valid is not None:
                    censored = ~v_reg[v_rank]
                    w = torch.where(
                        censored,
                        torch.full_like(w, self.censored_weight),
                        w,
                    )
                l_bpr = (F.softplus(-dy.sign() * ds) * w).sum() / w.sum().clamp(min=1e-6)

            task_losses.append(lam * (self.alpha * l_bpr + self.beta * l_reg))
            log[f"{name}_bpr"] = float(l_bpr.item())
            log[f"{name}_reg"] = float(l_reg.item())
            log[f"{name}_n_rank"] = int(v_rank.sum())
            log[f"{name}_n_reg"] = int(v_reg.sum())

        total = sum(task_losses)
        log["total"] = float(total.item())
        return total, log
