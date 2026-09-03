"""Model definitions: DMPNNEncoder + PolymerRankingModel."""

from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn

try:
    from chemprop.data.collate import BatchMolGraph
    from chemprop.nn import BondMessagePassing, MeanAggregation, SumAggregation, NormAggregation
except ImportError as e:
    raise ImportError(
        "chemprop>=2.0.0 is required. Install with: pip install chemprop>=2.0.0"
    ) from e

from .config import EXTRA_DIM, NUM_TASKS
from .featurizer import compute_edge_weights, get_feature_dims


class WeightedBondMessagePassing(BondMessagePassing):
    r"""D-MPNN with per-edge weights applied during message passing.

    Scales each directed edge's hidden representation by its edge weight before
    scatter-reduce aggregation, so that bonds touching sp3-hybridized carbons
    (non-conjugated side chains) contribute less to the learned representation:

    .. math::
        m_{vw}^{(t)} = \sum_{u \in \mathcal N(v)\setminus w}
                       w_{uv}\, h_{uv}^{(t-1)}

    where :math:`w_{uv}=0.2` if the bond *u–v* touches an sp3 carbon, else 1.0.

    Parameters
    ----------
    sp3_weight : float, default 0.2
        Edge weight for bonds whose source or destination atom is an sp3-hybridized carbon.
    """

    def __init__(self, *args, sp3_weight: float = 0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.sp3_weight = sp3_weight

    def _edge_weights(self, bmg: BatchMolGraph) -> Tensor:
        return compute_edge_weights(bmg.V, bmg.edge_index, bmg.E, self.sp3_weight)

    def message(self, H: Tensor, bmg: BatchMolGraph, edge_weights: Tensor) -> Tensor:
        H_w = H * edge_weights.unsqueeze(1)
        index_torch = bmg.edge_index[1].unsqueeze(1).repeat(1, H.shape[1])
        M_all = torch.zeros(
            len(bmg.V), H.shape[1], dtype=H.dtype, device=H.device
        ).scatter_reduce_(0, index_torch, H_w, reduce="sum", include_self=False)[
            bmg.edge_index[0]
        ]
        M_rev = H_w[bmg.rev_edge_index]
        return M_all - M_rev

    def forward(self, bmg: BatchMolGraph, V_d: Optional[Tensor] = None) -> Tensor:
        # Compute edge weights BEFORE graph_transform so that one-hot
        # comparisons (e.g. V[:, SP3_IDX] == 1.0) are not broken by scaling.
        edge_weights = self._edge_weights(bmg)
        bmg = self.graph_transform(bmg)

        H_0 = self.initialize(bmg)
        H = self.tau(H_0)
        for _ in range(1, self.depth):
            if self.undirected:
                H = (H + H[bmg.rev_edge_index]) / 2
            M = self.message(H, bmg, edge_weights)
            H = self.update(M, H_0)

        # final per-atom aggregation, also weighted
        H_final = H * edge_weights.unsqueeze(1)
        index_torch = bmg.edge_index[1].unsqueeze(1).repeat(1, H.shape[1])
        M = torch.zeros(
            len(bmg.V), H.shape[1], dtype=H.dtype, device=H.device
        ).scatter_reduce_(0, index_torch, H_final, reduce="sum", include_self=False)
        return self.finalize(M, bmg.V, V_d)


class DMPNNEncoder(nn.Module):
    """
    Directed Message Passing Neural Network (D-MPNN) encoder.
    Uses WeightedBondMessagePassing to down-weight sp3-carbon (non-conjugated)
    edges during message passing.
    Output: [B, hidden_size]
    """

    def __init__(self,
                 hidden_size: int = 300,
                 depth: int = 6,
                 dropout: float = 0.1,
                 aggregation: str = "mean",
                 d_v: Optional[int] = None,
                 d_e: Optional[int] = None,
                 sp3_weight: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.aggregation = aggregation
        self.sp3_weight = sp3_weight

        if d_v is None or d_e is None:
            d_v, d_e = get_feature_dims()

        self.mpnn = WeightedBondMessagePassing(
            d_v=d_v,
            d_e=d_e,
            d_h=hidden_size,
            depth=depth,
            dropout=dropout,
            sp3_weight=sp3_weight,
        )
        if aggregation == "mean":
            self.agg = MeanAggregation()
        elif aggregation == "sum":
            self.agg = SumAggregation()
        else:
            self.agg = NormAggregation()

    def forward(self, batch: BatchMolGraph) -> torch.Tensor:
        device = next(self.parameters()).device
        batch.to(device)
        H = self.mpnn(batch)
        mol_vecs = self.agg(H, batch.batch)
        return mol_vecs


class PolymerRankingModel(nn.Module):
    """
    Architecture:
        SMILES (multiple structures) ──► D-MPNN ──► avg embeddings ──► mol_emb [H]
                                                                          ├─ cat ──► FFN ──► [score_e, score_h]
        extra_features [5] ──────────────────────────────────────────────┘

    Two polymers share the same parameters (siamese network).
    Each polymer may have multiple cyclized structures (from monomer concatenation);
    their D-MPNN embeddings are averaged before being passed to the FFN.
    """

    def __init__(
        self,
        hidden_size: int = 300,
        depth: int = 6,
        dropout: float = 0.1,
        ffn_hidden: int = 256,
        extra_dim: int = EXTRA_DIM,
        num_tasks: int = NUM_TASKS,
        aggregation: str = "mean",
        sp3_weight: float = 0.2,
    ):
        super().__init__()
        self.mpnn = DMPNNEncoder(
            hidden_size, depth, dropout, aggregation, sp3_weight=sp3_weight
        )

        ffn_in = hidden_size + extra_dim
        self.ffn = nn.Sequential(
            nn.Linear(ffn_in, ffn_hidden),
            nn.LayerNorm(ffn_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, ffn_hidden // 2),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(ffn_hidden // 2, num_tasks),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _average_by_group(emb: torch.Tensor,
                          group_sizes: List[int]) -> torch.Tensor:
        """
        Average embeddings by group.

        Parameters
        ----------
        emb : Tensor [total_structures, H]
            Concatenated embeddings for all structures across all samples.
        group_sizes : list[int]
            Number of structures per sample, length = batch_size.

        Returns
        -------
        Tensor [batch_size, H]
            Averaged embedding per sample.
        """
        result = []
        offset = 0
        for n in group_sizes:
            result.append(emb[offset:offset + n].mean(dim=0))
            offset += n
        return torch.stack(result)

    def encode(
        self,
        mol_graphs: BatchMolGraph,
        n_structures: List[int],
        extra: torch.Tensor,
    ) -> torch.Tensor:
        """Returns [B, num_tasks] score vector."""
        emb = self.mpnn(mol_graphs)
        emb = self._average_by_group(emb, n_structures)
        x = torch.cat([emb, extra], dim=-1)
        return self.ffn(x)

    def forward(
        self,
        mg1: BatchMolGraph,
        n1: List[int],
        ef1: torch.Tensor,
        mg2: BatchMolGraph,
        n2: List[int],
        ef2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (scores_1, scores_2), each [B, T]."""
        return self.encode(mg1, n1, ef1), self.encode(mg2, n2, ef2)
