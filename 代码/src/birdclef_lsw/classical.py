from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import warnings

from birdclef_step3 import Step3Config
from birdclef_step3.data import load_step3_dataset

from . import OWNER_NAME
from .metrics import LSW_METRIC_COLUMNS, compute_lsw_metrics, find_best_global_threshold


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def build_lsw_classical_models(random_seed: int) -> dict[str, object]:
    return {
        "extra_trees": OneVsRestClassifier(
            ExtraTreesClassifier(
                n_estimators=260,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=random_seed,
                n_jobs=-1,
            )
        ),
        "linear_svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    OneVsRestClassifier(
                        LinearSVC(class_weight="balanced", random_state=random_seed, dual="auto", max_iter=6000)
                    ),
                ),
            ]
        ),
        "knn_distance": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", OneVsRestClassifier(KNeighborsClassifier(n_neighbors=7, weights="distance"))),
            ]
        ),
        "logistic_regression_l1": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    OneVsRestClassifier(
                        LogisticRegression(
                            penalty="l1",
                            solver="liblinear",
                            class_weight="balanced",
                            max_iter=2500,
                            random_state=random_seed,
                        )
                    ),
                ),
            ]
        ),
    }


def _ensure_2d(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _score_model(model: object, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return _ensure_2d(model.predict_proba(X))
    if hasattr(model, "decision_function"):
        return _sigmoid(_ensure_2d(model.decision_function(X)))
    return _ensure_2d(model.predict(X)).astype(np.float64)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _plot_model_summary(summary_df: pd.DataFrame, output_path: Path) -> None:
    if summary_df.empty:
        return
    plot_df = summary_df.sort_values("lrap", ascending=False).copy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    specs = [
        ("lrap", "LRAP (higher is better)", "#2a9d8f"),
        ("micro_f1", "Micro F1 (higher is better)", "#457b9d"),
        ("hamming_loss", "Hamming Loss (lower is better)", "#e76f51"),
    ]
    for ax, (metric, title, color) in zip(axes, specs):
        ax.bar(plot_df["model"], plot_df[metric], color=color)
        ax.set_title(title)
        ax.set_ylim(0.0, max(1.0, float(plot_df[metric].max()) * 1.1))
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        for tick in ax.get_xticklabels():
            tick.set_rotation(20)
            tick.set_ha("right")
    fig.suptitle("Traditional Machine Learning Model Comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_thresholds(threshold_df: pd.DataFrame, output_path: Path) -> None:
    if threshold_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(threshold_df["model"], threshold_df["threshold"], color="#6d597a")
    ax.set_title("Global Threshold Optimization")
    ax.set_ylabel("Best threshold")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    for tick in ax.get_xticklabels():
        tick.set_rotation(20)
        tick.set_ha("right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_lsw_classical_experiments(workspace_root: Path, lsw_root: Path) -> dict[str, object]:
    workspace_root = workspace_root.resolve()
    lsw_root = lsw_root.resolve()
    artifacts_root = lsw_root / "报告" / "实验产物"
    tables_dir = artifacts_root / "tables"
    figures_dir = artifacts_root / "figures"
    models_dir = artifacts_root / "models"
    for path in (tables_dir, figures_dir, models_dir):
        path.mkdir(parents=True, exist_ok=True)

    config = Step3Config.from_project_root(workspace_root)
    X, Y, feature_columns, label_columns, window_ids, _, dataset_info = load_step3_dataset(config)
    X_values = X.reset_index(drop=True)
    Y_values = Y.reset_index(drop=True)
    y_array = Y_values.to_numpy(dtype=np.int64)
    splitter = KFold(n_splits=config.n_splits, shuffle=True, random_state=config.random_seed)
    models = build_lsw_classical_models(config.random_seed)

    fold_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    oof_frames: list[pd.DataFrame] = [pd.DataFrame({"window_id": window_ids})]
    best_model_name = ""
    best_lrap = -1.0

    for model_name, base_model in models.items():
        start = time.perf_counter()
        oof_scores = np.zeros((len(X_values), len(label_columns)), dtype=np.float64)
        model_fold_rows: list[dict[str, object]] = []
        for fold_index, (train_idx, valid_idx) in enumerate(splitter.split(X_values), start=1):
            model = clone(base_model)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                warnings.filterwarnings("ignore", message="Label .* is present in all training examples.")
                model.fit(X_values.iloc[train_idx], Y_values.iloc[train_idx])
            scores = _score_model(model, X_values.iloc[valid_idx])
            oof_scores[valid_idx, :] = scores
            metrics = compute_lsw_metrics(y_array[valid_idx], scores, threshold=0.5)
            row = {"model": model_name, "fold": fold_index, **metrics}
            fold_rows.append(row)
            model_fold_rows.append(row)
            print(
                f"[LSW classical {model_name} fold {fold_index}] "
                f"LRAP={metrics['lrap']:.4f} microF1={metrics['micro_f1']:.4f} top3={metrics['top3_hit_rate']:.4f}"
            )

        elapsed = time.perf_counter() - start
        threshold = find_best_global_threshold(y_array, oof_scores)
        threshold_rows.append({"model": model_name, **threshold})
        summary = {"model": model_name, "fit_seconds": round(elapsed, 3), "best_threshold": threshold["threshold"]}
        for metric in LSW_METRIC_COLUMNS:
            values = [float(row[metric]) for row in model_fold_rows]
            summary[metric] = float(np.mean(values)) if values else 0.0
            summary[f"{metric}_std"] = float(np.std(values)) if values else 0.0
        summary_rows.append(summary)
        oof_frames.append(pd.DataFrame(oof_scores, columns=[f"{model_name}__{label}" for label in label_columns]))
        if float(summary["lrap"]) > best_lrap:
            best_lrap = float(summary["lrap"])
            best_model_name = model_name

    best_model = build_lsw_classical_models(config.random_seed)[best_model_name]
    best_model.fit(X_values, Y_values)
    joblib.dump(
        {
            "model": best_model,
            "model_name": best_model_name,
            "feature_columns": feature_columns,
            "label_columns": label_columns,
        },
        models_dir / "best_classical_model.joblib",
    )

    summary_df = pd.DataFrame(summary_rows).sort_values("lrap", ascending=False)
    threshold_df = pd.DataFrame(threshold_rows)
    fold_df = pd.DataFrame(fold_rows)
    oof_df = pd.concat(oof_frames, axis=1)

    summary_df.to_csv(tables_dir / "lsw_model_summary.csv", index=False)
    summary_df.to_csv(tables_dir / "lsw_classical_summary.csv", index=False)
    fold_df.to_csv(tables_dir / "lsw_classical_fold_results.csv", index=False)
    threshold_df.to_csv(tables_dir / "lsw_threshold_optimization.csv", index=False)
    oof_df.to_csv(tables_dir / "lsw_classical_oof_predictions.csv", index=False)
    _plot_model_summary(summary_df, figures_dir / "lsw_classical_model_comparison.png")
    _plot_thresholds(threshold_df, figures_dir / "lsw_threshold_optimization.png")

    manifest = {
        "owner": OWNER_NAME,
        "dataset": "BirdCLEF2026",
        "status": "classical_completed",
        "generated_at": _now_iso(),
        "stage": "lsw_classical",
        "models": list(models.keys()),
        "best_classical_model": best_model_name,
        "sample_count": int(dataset_info["merged_sample_count"]),
        "feature_column_count": len(feature_columns),
        "label_column_count": len(label_columns),
        "metric_focus": ["LRAP", "label ranking loss", "coverage error", "hamming loss", "micro/macro F1", "Top-k hit rate"],
    }
    _write_json(tables_dir / "lsw_experiment_manifest.json", manifest)
    return manifest
