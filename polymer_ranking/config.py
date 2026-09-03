"""Constants and configuration dataclasses."""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any

# Constants
EXTRA_COLS = [
    "conjugation_{s}",
    "Isomer_{s}",
    "CentroSymmetry_{s}",
    "E_LUMO (eV)_{s}",
    "E_HOMO (eV)_{s}",
]
EXTRA_DIM = len(EXTRA_COLS)
TASK_NAMES = ["mu_e", "mu_h"]
NUM_TASKS = len(TASK_NAMES)
CP_FEATURE_DIM = 1


# Configuration dataclasses
@dataclass
class ModelConfig:
    """Model architecture configuration"""
    hidden_size: int = 300
    depth: int = 6
    dropout: float = 0.1
    ffn_hidden: int = 256
    extra_dim: int = EXTRA_DIM
    num_tasks: int = NUM_TASKS
    aggregation: str = "mean"
    sp3_weight: float = 0.2

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        return cls(**{
            k: v
            for k, v in d.items() if k in cls.__dataclass_fields__
        })


@dataclass
class TrainingConfig:
    """Training configuration"""
    csv_path: str = "contrastive_paired.csv"
    save_dir: str = "checkpoints"
    epochs: int = 1000
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 50
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42

    def __post_init__(self):
        if not 0 < self.test_ratio < 1:
            raise ValueError(f"test_ratio must be in (0, 1), got {self.test_ratio}")
        if not 0 <= self.val_ratio < 1 - self.test_ratio:
            raise ValueError(
                f"val_ratio must be in [0, {1 - self.test_ratio:.2f}), got {self.val_ratio}"
            )


@dataclass
class FinetuneConfig:
    """Fine-tuning configuration"""
    csv_path: str = "contrastive_paired.csv"
    checkpoint_path: str = "checkpoints/best_model.pt"
    save_dir: str = "checkpoints"
    finetune_epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-5
    weight_decay: float = 1e-6
    seed: int = 42


@dataclass
class PredictConfig:
    """Prediction configuration"""
    predict_csv: str = ""
    checkpoint_path: str = "checkpoints/best_model.pt"
    output_path: str = "predictions.csv"
    batch_size: int = 32
