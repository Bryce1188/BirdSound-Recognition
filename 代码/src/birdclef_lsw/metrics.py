from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from sklearn.metrics import (
    coverage_error,
    f1_score,
    hamming_loss,
    label_ranking_average_precision_score,
    label_ranking_loss,
)


def _as_binary_matrix(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D matrix, got shape {array.shape}.")
    return (array > 0).astype(np.int64)


def _as_score_matrix(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D score matrix, got shape {array.shape}.")
    if not np.all(np.isfinite(array)):
        array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
    return array


def _topk_hit_rate(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive.")
    positives = np.sum(y_true, axis=1) > 0
    if not np.any(positives):
        return 0.0
    k = min(k, y_score.shape[1])
    top_indices = np.argpartition(-y_score, kth=k - 1, axis=1)[:, :k]
    hits = []
    for row_index, indices in enumerate(top_indices):
        if not positives[row_index]:
            continue
        hits.append(bool(np.any(y_true[row_index, indices] > 0)))
    return float(np.mean(hits)) if hits else 0.0


def compute_lsw_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float = 0.5,
    top_k: Iterable[int] = (1, 3),
) -> dict[str, float]:
    """Compute the ranking-first metric set used by the LSW version."""
    y_true_bin = _as_binary_matrix(y_true)
    y_score_arr = _as_score_matrix(y_score)
    if y_true_bin.shape != y_score_arr.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true_bin.shape}, y_score={y_score_arr.shape}.")

    y_pred = (y_score_arr >= float(threshold)).astype(np.int64)
    payload = {
        "lrap": float(label_ranking_average_precision_score(y_true_bin, y_score_arr)),
        "label_ranking_loss": float(label_ranking_loss(y_true_bin, y_score_arr)),
        "coverage_error": float(coverage_error(y_true_bin, y_score_arr)),
        "hamming_loss": float(hamming_loss(y_true_bin, y_pred)),
        "micro_f1": float(f1_score(y_true_bin, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true_bin, y_pred, average="macro", zero_division=0)),
        "prediction_density": float(np.mean(y_pred)),
    }
    for k in top_k:
        payload[f"top{int(k)}_hit_rate"] = _topk_hit_rate(y_true_bin, y_score_arr, int(k))
    return payload


def find_best_global_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    candidate_thresholds: Iterable[float] | None = None,
) -> dict[str, float]:
    thresholds = list(candidate_thresholds or np.round(np.arange(0.05, 0.96, 0.05), 2))
    best: dict[str, float] | None = None
    for threshold in thresholds:
        metrics = compute_lsw_metrics(y_true, y_score, threshold=float(threshold))
        row = {"threshold": float(threshold), **metrics}
        if best is None:
            best = row
            continue
        if row["micro_f1"] > best["micro_f1"] + 1e-12:
            best = row
        elif abs(row["micro_f1"] - best["micro_f1"]) <= 1e-12 and row["hamming_loss"] < best["hamming_loss"]:
            best = row
    if best is None:
        raise ValueError("No candidate thresholds were provided.")
    return best


LSW_METRIC_COLUMNS = [
    "lrap",
    "label_ranking_loss",
    "coverage_error",
    "hamming_loss",
    "micro_f1",
    "macro_f1",
    "top1_hit_rate",
    "top3_hit_rate",
    "prediction_density",
]
