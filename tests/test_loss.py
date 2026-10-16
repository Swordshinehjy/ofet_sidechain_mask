"""Tests for MultiTaskBayesianRankingLoss (ranking + delta regression).

All score/target tensors are [B, 2] (one column per task: mu_e, mu_h).
"""

import math

import pytest
import torch

from polymer_ranking.loss import MultiTaskBayesianRankingLoss


def _bpr(dy_sign, ds):
    return math.log(1 + math.exp(-dy_sign * ds))  # softplus(-sign(dy)*ds)


# ── basic behavior ──────────────────────────────────────────────────────────

def test_loss_is_finite_and_positive():
    torch.manual_seed(0)
    s1, s2 = torch.randn(4, 2), torch.randn(4, 2)
    y1, y2 = torch.randn(4, 2), torch.randn(4, 2)
    loss, log = MultiTaskBayesianRankingLoss()(s1, s2, y1, y2)
    assert torch.isfinite(loss)
    assert loss.item() > 0
    assert log["total"] == pytest.approx(loss.item())


def test_correct_ranking_costs_less_than_wrong():
    # y1 > y2 for both tasks
    y1 = torch.tensor([[1.0, 1.0]])
    y2 = torch.tensor([[0.0, 0.0]])
    good = MultiTaskBayesianRankingLoss()(
        torch.tensor([[5.0, 5.0]]), torch.tensor([[0.0, 0.0]]), y1, y2)
    bad = MultiTaskBayesianRankingLoss()(
        torch.tensor([[0.0, 0.0]]), torch.tensor([[5.0, 5.0]]), y1, y2)
    assert good[0].item() < bad[0].item()


def test_bpr_term_decreases_with_score_margin():
    y1 = torch.tensor([[2.0, 2.0]])
    y2 = torch.tensor([[1.0, 1.0]])
    near = MultiTaskBayesianRankingLoss(rank_weight=1.0, reg_weight=0.0)(
        torch.tensor([[1.5, 1.5]]), torch.tensor([[1.0, 1.0]]), y1, y2)
    far = MultiTaskBayesianRankingLoss(rank_weight=1.0, reg_weight=0.0)(
        torch.tensor([[5.0, 5.0]]), torch.tensor([[1.0, 1.0]]), y1, y2)
    assert far[0].item() < near[0].item()


# ── delta regression term ───────────────────────────────────────────────────

def test_regression_target_is_scaled_by_delta_scale():
    # task 0: s1-s2 = 2, y1-y2 = 1.5, scale=3 -> target 0.5 -> mse = 2.25
    # task 1: dy = 0 and ds = 0 -> contributes nothing
    crit = MultiTaskBayesianRankingLoss(rank_weight=0.0, reg_weight=1.0,
                                        delta_scale=3.0)
    loss, log = crit(torch.tensor([[3.0, 1.0]]), torch.tensor([[1.0, 1.0]]),
                     torch.tensor([[2.0, 0.0]]), torch.tensor([[0.5, 0.0]]))
    assert log["mu_e_reg"] == pytest.approx(2.25)
    assert log["mu_h_reg"] == pytest.approx(0.0)
    assert loss.item() == pytest.approx(2.25)


def test_regression_only_on_valid_pairs():
    # pair 0: both measured (dy known); pair 1: censored, must not hit regression
    # masks are [pair, task]
    s1 = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    s2 = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    y1 = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    y2 = torch.tensor([[0.5, 0.5], [0.0, 0.0]])
    valid = torch.tensor([[True, True], [False, False]])
    rank_valid = torch.tensor([[True, True], [True, True]])

    crit = MultiTaskBayesianRankingLoss(rank_weight=0.0, reg_weight=1.0,
                                        delta_scale=1.0)
    loss, log = crit(s1, s2, y1, y2, valid=valid, rank_valid=rank_valid)
    assert log["mu_e_n_reg"] == 1
    assert log["mu_e_reg"] == pytest.approx(0.25)


def test_zero_valid_pairs_give_zero_terms():
    crit = MultiTaskBayesianRankingLoss()
    s = torch.zeros(2, 2)
    y = torch.zeros(2, 2)
    valid = torch.zeros(2, 2, dtype=torch.bool)
    loss, log = crit(s, s, y, y, valid=valid, rank_valid=valid)
    assert loss.item() == 0.0
    assert log["mu_e_bpr"] == 0.0
    assert log["mu_e_reg"] == 0.0


# ── BPR term and masks ──────────────────────────────────────────────────────

def test_bpr_uses_rank_valid_mask():
    # pair 1 has y1 == y2 -> ordering undecidable -> excluded from BPR
    s1 = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    s2 = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    y1 = torch.tensor([[2.0, 2.0], [1.0, 1.0]])
    y2 = torch.tensor([[1.0, 1.0], [1.0, 1.0]])

    crit = MultiTaskBayesianRankingLoss(rank_weight=1.0, reg_weight=0.0)
    loss, log = crit(s1, s2, y1, y2)
    assert log["mu_e_n_rank"] == 1
    assert log["mu_e_bpr"] == pytest.approx(_bpr(1.0, 1.0))


def test_censored_pair_gets_downweighted():
    # two-pair batch: pair 0 measured, pair 1 censored (y2 = 0)
    # score margins differ (1.0 vs 2.0) so the 0.5 weight actually shows up
    s1 = torch.tensor([[2.0, 2.0], [2.0, 2.0]])
    s2 = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    y1 = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    y2 = torch.tensor([[0.5, 0.5], [0.0, 0.0]])
    valid = torch.tensor([[True, True], [False, False]])      # pair 1 censored
    rank_valid = torch.tensor([[True, True], [True, True]])   # ordering decidable

    sp = lambda x: math.log(1 + math.exp(-x))  # softplus(-x)

    weighted = MultiTaskBayesianRankingLoss(rank_weight=1.0, reg_weight=0.0,
                                            censored_weight=0.5)
    _, log_w = weighted(s1, s2, y1, y2, valid=valid, rank_valid=rank_valid)
    assert log_w["mu_e_bpr"] == pytest.approx(
        (1.0 * sp(1.0) + 0.5 * sp(2.0)) / 1.5)

    full = MultiTaskBayesianRankingLoss(rank_weight=1.0, reg_weight=0.0,
                                        censored_weight=1.0)
    _, log_f = full(s1, s2, y1, y2, valid=valid, rank_valid=rank_valid)
    assert log_f["mu_e_bpr"] == pytest.approx((sp(1.0) + sp(2.0)) / 2)

    assert log_w["mu_e_bpr"] != pytest.approx(log_f["mu_e_bpr"])


def test_censored_pair_excluded_from_regression():
    s1 = torch.tensor([[2.0, 2.0]])
    s2 = torch.tensor([[0.0, 0.0]])
    y1 = torch.tensor([[1.0, 1.0]])
    y2 = torch.tensor([[0.0, 0.0]])
    valid = torch.tensor([[False, False]])
    rank_valid = torch.tensor([[True, True]])

    crit = MultiTaskBayesianRankingLoss(rank_weight=0.0, reg_weight=1.0)
    _, log = crit(s1, s2, y1, y2, valid=valid, rank_valid=rank_valid)
    assert log["mu_e_n_reg"] == 0
    assert log["mu_e_reg"] == 0.0


def test_no_masks_defaults_to_all_measured():
    # valid=None -> every pair is treated as measured
    s1 = torch.tensor([[2.0, 2.0], [0.0, 0.0]])
    s2 = torch.tensor([[0.0, 0.0], [2.0, 2.0]])
    y1 = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    y2 = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

    crit = MultiTaskBayesianRankingLoss(rank_weight=1.0, reg_weight=1.0,
                                        delta_scale=1.0)
    loss, log = crit(s1, s2, y1, y2)
    assert log["mu_e_n_rank"] == 2
    assert log["mu_e_n_reg"] == 2
    assert log["mu_e_reg"] == pytest.approx(1.0)  # (2-1)^2 and (-2+1)^2


# ── task weighting and logging ──────────────────────────────────────────────

def test_task_weights_scale_per_task_contributions():
    # both tasks get identical inputs -> identical per-task losses
    s1 = torch.tensor([[2.0, 2.0]])
    s2 = torch.tensor([[0.0, 0.0]])
    y1 = torch.tensor([[1.0, 1.0]])
    y2 = torch.tensor([[0.0, 0.0]])

    uniform = MultiTaskBayesianRankingLoss(rank_weight=1.0, reg_weight=0.0)
    weighted = MultiTaskBayesianRankingLoss(rank_weight=1.0, reg_weight=0.0,
                                            task_weights=[2.0, 1.0])
    _, log_u = uniform(s1, s2, y1, y2)
    _, log_w = weighted(s1, s2, y1, y2)
    # uniform total = 1*L + 1*L = 2L; weighted total = 2*L + 1*L = 3L
    assert log_w["total"] == pytest.approx(1.5 * log_u["total"])
    assert log_u["mu_e_bpr"] == pytest.approx(log_u["mu_h_bpr"])


def test_log_dict_contains_all_keys():
    crit = MultiTaskBayesianRankingLoss()
    _, log = crit(torch.randn(3, 2), torch.randn(3, 2),
                  torch.randn(3, 2), torch.randn(3, 2))
    for key in ("mu_e_bpr", "mu_e_reg", "mu_e_n_rank", "mu_e_n_reg",
                "mu_h_bpr", "mu_h_reg", "mu_h_n_rank", "mu_h_n_reg", "total"):
        assert key in log


def test_invalid_delta_scale_falls_back_to_one():
    assert MultiTaskBayesianRankingLoss(delta_scale=0.0).delta_scales == [1.0, 1.0]
    assert MultiTaskBayesianRankingLoss(delta_scale=2.5).delta_scales == [2.5, 2.5]


def test_delta_scale_is_per_task():
    """A sequence is kept per task; a scalar is broadcast to every task."""
    crit = MultiTaskBayesianRankingLoss(rank_weight=0.0, reg_weight=1.0,
                                        delta_scale=[0.5, 3.0])
    assert crit.delta_scales == [0.5, 3.0]

    # ds = 2.0, dy = 1.5 for both tasks
    # mu_e: target = 1.5 / 0.5 = 3.0 -> mse = (2 - 3)^2   = 1.0
    # mu_h: target = 1.5 / 3.0 = 0.5 -> mse = (2 - 0.5)^2 = 2.25
    _, log = crit(torch.tensor([[2.0, 2.0]]), torch.tensor([[0.0, 0.0]]),
                  torch.tensor([[1.5, 1.5]]), torch.tensor([[0.0, 0.0]]))
    assert log["mu_e_reg"] == pytest.approx(1.0)
    assert log["mu_h_reg"] == pytest.approx(2.25)


def test_delta_scale_rejects_wrong_length():
    with pytest.raises(ValueError, match="delta_scale"):
        MultiTaskBayesianRankingLoss(delta_scale=[1.0, 2.0, 3.0])
