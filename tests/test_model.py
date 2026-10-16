"""Tests for DMPNNEncoder, PolymerRankingModel, and edge-weighted message passing."""

import pytest
import torch
from chemprop.data.collate import BatchMolGraph
from rdkit import Chem

from polymer_ranking.config import EXTRA_DIM, NUM_TASKS
from polymer_ranking.featurizer import create_featurizer
from polymer_ranking.model import (
    DMPNNEncoder,
    PolymerRankingModel,
    WeightedBondMessagePassing,
)

from samples import ALKYL_THIOPHENE, THIOPHENE, BENZENE
from conftest import TINY_MODEL_KWARGS


def _batch_molgraph(smiles_list):
    feat = create_featurizer()
    graphs = [feat(Chem.MolFromSmiles(s)) for s in smiles_list]
    return BatchMolGraph(graphs), graphs


# ── _average_by_group ───────────────────────────────────────────────────────

def test_average_by_group_manual():
    emb = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    out = PolymerRankingModel._average_by_group(emb, [2, 1])
    assert out.shape == (2, 2)
    torch.testing.assert_close(out, torch.tensor([[2.0, 3.0], [5.0, 6.0]]))


def test_average_by_group_singletons_equal_input():
    emb = torch.randn(4, 8)
    out = PolymerRankingModel._average_by_group(emb, [1, 1, 1, 1])
    torch.testing.assert_close(out, emb)


# ── DMPNNEncoder ────────────────────────────────────────────────────────────

def test_encoder_output_shape(tiny_model):
    encoder = tiny_model.mpnn
    bmg, _ = _batch_molgraph([THIOPHENE, BENZENE])
    with torch.no_grad():
        out = encoder(bmg)
    assert out.shape == (2, encoder.hidden_size)


@pytest.mark.parametrize("aggregation", ["mean", "sum", "norm"])
def test_encoder_aggregation_variants(aggregation):
    encoder = DMPNNEncoder(hidden_size=16, depth=2, aggregation=aggregation)
    encoder.eval()
    bmg, _ = _batch_molgraph([THIOPHENE])
    with torch.no_grad():
        out = encoder(bmg)
    assert out.shape == (1, 16)


def test_encoder_forward_batch_invariance():
    """Encoding a molecule alone vs in a batch must give the same embedding."""
    encoder = DMPNNEncoder(hidden_size=16, depth=2)
    encoder.eval()
    alone, _ = _batch_molgraph([THIOPHENE])
    with_benzene, _ = _batch_molgraph([BENZENE, THIOPHENE])
    with torch.no_grad():
        a = encoder(alone)
        b = encoder(with_benzene)
    torch.testing.assert_close(a[0], b[1], rtol=1e-4, atol=1e-5)


def test_encoder_stores_sp3_weight():
    encoder = DMPNNEncoder(hidden_size=16, depth=2, sp3_weight=0.35)
    assert encoder.sp3_weight == 0.35
    assert encoder.mpnn.sp3_weight == 0.35


def test_weighted_mp_matches_chemprop_when_weight_is_one():
    """Guard for the chemprop internals we re-implement in `forward`.

    With sp3_weight=1.0 no edge is down-weighted, so our forward must be
    numerically identical to the upstream BondMessagePassing forward. If this
    breaks after a chemprop upgrade, the private API we rely on has changed.
    """
    from chemprop.nn import BondMessagePassing as UpstreamBondMessagePassing
    from polymer_ranking.featurizer import get_feature_dims

    d_v, d_e = get_feature_dims()
    ours = WeightedBondMessagePassing(d_v=d_v, d_e=d_e, d_h=16, depth=3,
                                      sp3_weight=1.0)
    upstream = UpstreamBondMessagePassing(d_v=d_v, d_e=d_e, d_h=16, depth=3)
    upstream.load_state_dict(ours.state_dict())
    ours.eval()
    upstream.eval()

    bmg, _ = _batch_molgraph([ALKYL_THIOPHENE, THIOPHENE])
    with torch.no_grad():
        torch.testing.assert_close(ours(bmg), upstream(bmg))


def test_weighted_message_passing_uses_edge_weights():
    from polymer_ranking.featurizer import get_feature_dims

    d_v, d_e = get_feature_dims()
    mp = WeightedBondMessagePassing(d_v=d_v, d_e=d_e, d_h=16, depth=2,
                                    sp3_weight=0.2)
    bmg, _ = _batch_molgraph([ALKYL_THIOPHENE])
    weights = mp._edge_weights(bmg)
    assert torch.any(weights == 0.2) and torch.any(weights == 1.0)


# ── PolymerRankingModel ─────────────────────────────────────────────────────

def test_model_forward_shapes(tiny_model):
    bmg1, g1s = _batch_molgraph([THIOPHENE, BENZENE])
    bmg2, g2s = _batch_molgraph([BENZENE])
    ef1 = torch.zeros(2, EXTRA_DIM)
    ef2 = torch.zeros(1, EXTRA_DIM)
    with torch.no_grad():
        s1, s2 = tiny_model(bmg1, [1, 1], ef1, bmg2, [1], ef2)
    assert s1.shape == (2, NUM_TASKS)
    assert s2.shape == (1, NUM_TASKS)


def test_model_multi_structure_averaging(tiny_model):
    """Multi-structure samples: raw D-MPNN embeddings are averaged BEFORE the FFN."""
    tiny_model.eval()
    bmg_two_mols, _ = _batch_molgraph([THIOPHENE, BENZENE])
    ef = torch.zeros(1, EXTRA_DIM)
    with torch.no_grad():
        emb = tiny_model.mpnn(bmg_two_mols)                    # [2, H]
        emb_avg = PolymerRankingModel._average_by_group(emb, [2])
        expected = tiny_model.ffn(torch.cat([emb_avg, ef], dim=-1))

        s_avg = tiny_model.encode(bmg_two_mols, [2], ef)
    torch.testing.assert_close(s_avg, expected)
    # pre-FFN averaging differs from post-FFN averaging
    s_separate = tiny_model.encode(bmg_two_mols, [1, 1], torch.zeros(2, EXTRA_DIM))
    assert not torch.allclose(s_avg, s_separate.mean(dim=0, keepdim=True))


def test_model_siamese_shared_encoder(tiny_model):
    # siamese architecture: exactly ONE D-MPNN encoder shared by both sides
    encoders = [m for m in tiny_model.modules() if isinstance(m, DMPNNEncoder)]
    assert len(encoders) == 1
    ffns = [m for m in tiny_model.modules() if isinstance(m, PolymerRankingModel)]
    assert len(ffns) == 1


def test_model_eval_determinism(tiny_model):
    bmg, _ = _batch_molgraph([THIOPHENE, ALKYL_THIOPHENE])
    ef1 = torch.randn(2, EXTRA_DIM)
    ef2 = torch.randn(1, EXTRA_DIM)
    with torch.no_grad():
        s1a, s2a = tiny_model(bmg, [1, 1], ef1, bmg, [2], ef2)
        s1b, s2b = tiny_model(bmg, [1, 1], ef1, bmg, [2], ef2)
    torch.testing.assert_close(s1a, s1b)
    torch.testing.assert_close(s2a, s2b)


def test_model_extra_dim_wiring():
    model = PolymerRankingModel(**TINY_MODEL_KWARGS, extra_dim=3)
    assert model.ffn[0].in_features == TINY_MODEL_KWARGS["hidden_size"] + 3


def test_model_output_dim_equals_num_tasks():
    model = PolymerRankingModel(**TINY_MODEL_KWARGS, num_tasks=4)
    assert model.ffn[-1].out_features == 4


def test_model_xavier_init_applied(tiny_model):
    # _init_weights zeroes biases of the FFN linear layers
    for m in tiny_model.ffn.modules():
        if isinstance(m, torch.nn.Linear):
            assert torch.all(m.bias == 0)
