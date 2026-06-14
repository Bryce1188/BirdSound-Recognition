from __future__ import annotations

from .config import Step4Config


def load_step4_dataset(config: Step4Config):
    raise FileNotFoundError(
        "The original Step4 dataset loader was not included in this handoff. "
        "The web demo can still load the saved checkpoints, but retraining the "
        "deep models requires the full Step4 data package."
    )
