from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Step4Config:
    project_root: Path
    sample_rate: int = 32000
    duration_seconds: float = 5.0
    n_fft: int = 2048
    hop_length: int = 512
    win_length: int = 2048
    n_mels: int = 128
    f_min: int = 50
    f_max: int = 14000
    top_db: int = 80
    dropout: float = 0.35
    n_splits: int = 5
    batch_size: int = 48
    max_epochs: int = 10
    early_stopping_patience: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    random_seed: int = 42

    @classmethod
    def from_project_root(cls, project_root: Path) -> "Step4Config":
        return cls(project_root=Path(project_root))
