from __future__ import annotations

import json
import os
import random
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from birdclef_step4.config import Step4Config
from birdclef_step4.data import load_step4_dataset
from birdclef_step4.model import BirdCLEFCNN

from . import OWNER_NAME
from .metrics import LSW_METRIC_COLUMNS, compute_lsw_metrics, find_best_global_threshold


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class LswSpectrogramDataset(Dataset):
    def __init__(self, spectrograms: np.ndarray, labels: np.ndarray, indices: np.ndarray, spec_augment: bool = False) -> None:
        self.spectrograms = spectrograms
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.spec_augment = spec_augment

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = int(self.indices[item])
        spec = np.clip((self.spectrograms[index] + 80.0) / 80.0, 0.0, 1.0).astype(np.float32, copy=True)
        if self.spec_augment:
            spec = _apply_simple_specaugment(spec)
        label = self.labels[index].astype(np.float32, copy=False)
        return torch.from_numpy(spec[None, :, :]), torch.from_numpy(label)


def _apply_simple_specaugment(spec: np.ndarray) -> np.ndarray:
    output = spec.copy()
    if output.shape[0] > 12:
        width = min(10, max(1, output.shape[0] // 12))
        start = random.randint(0, output.shape[0] - width)
        output[start : start + width, :] = float(np.mean(output))
    if output.shape[1] > 20:
        width = min(18, max(1, output.shape[1] // 14))
        start = random.randint(0, output.shape[1] - width)
        output[:, start : start + width] = float(np.mean(output))
    return output.astype(np.float32, copy=False)


class AttentionBiGRU(nn.Module):
    def __init__(self, num_labels: int, dropout: float = 0.35, hidden_size: int = 64) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 48)),
        )
        self.gru = nn.GRU(96, hidden_size, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(nn.Linear(hidden_size * 2, 64), nn.Tanh(), nn.Linear(64, 1))
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size * 2, num_labels))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.cnn(inputs).squeeze(2).transpose(1, 2)
        sequence, _ = self.gru(features)
        weights = torch.softmax(self.attention(sequence), dim=1)
        pooled = torch.sum(sequence * weights, dim=1)
        return self.classifier(pooled)


def build_lsw_deep_model(model_name: str, num_labels: int, dropout: float) -> nn.Module:
    if model_name == "cnn_baseline_lsw":
        return BirdCLEFCNN(num_labels=num_labels, dropout=dropout)
    if model_name == "attention_bigru":
        return AttentionBiGRU(num_labels=num_labels, dropout=dropout)
    raise ValueError(f"Unknown LSW deep model: {model_name}")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pos_weight(labels: np.ndarray) -> torch.Tensor:
    positives = np.sum(labels, axis=0).astype(np.float32)
    negatives = labels.shape[0] - positives
    weights = np.ones_like(positives, dtype=np.float32)
    valid = positives > 0
    weights[valid] = negatives[valid] / positives[valid]
    return torch.tensor(weights, dtype=torch.float32)


def _evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses: list[float] = []
    scores: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    use_amp = device.type == "cuda"
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.float32)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(inputs)
                loss = criterion(logits, labels)
            losses.append(float(loss.item()))
            scores.append(torch.sigmoid(logits).detach().cpu().numpy())
            targets.append(labels.detach().cpu().numpy())
    return float(np.mean(losses)) if losses else 0.0, np.vstack(scores), np.vstack(targets)


def _save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    model_name: str,
    fold: int,
    best_epoch: int,
    label_columns: list[str],
    config: Step4Config,
) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "model_name": model_name,
        "fold": fold,
        "best_epoch": best_epoch,
        "label_columns": label_columns,
        "config": {
            "architecture": model_name,
            "sample_rate": config.sample_rate,
            "duration_seconds": config.duration_seconds,
            "n_fft": config.n_fft,
            "hop_length": config.hop_length,
            "win_length": config.win_length,
            "n_mels": config.n_mels,
            "f_min": config.f_min,
            "f_max": config.f_max,
            "top_db": config.top_db,
            "dropout": config.dropout,
        },
    }
    torch.save(payload, path)


def _plot_deep_summary(summary_df: pd.DataFrame, output_path: Path) -> None:
    if summary_df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric, title, color in [
        (axes[0], "lrap", "LRAP", "#2a9d8f"),
        (axes[1], "top3_hit_rate", "Top-3 Hit Rate", "#457b9d"),
        (axes[2], "micro_f1", "Micro F1", "#e76f51"),
    ]:
        ax.bar(summary_df["model"], summary_df[metric], color=color)
        ax.set_title(title)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
    fig.suptitle("Deep Learning Model Comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_epoch_history(history_df: pd.DataFrame, output_path: Path) -> None:
    if history_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for model_name, model_df in history_df.groupby("model"):
        grouped = model_df.groupby("epoch", as_index=False)[["train_loss", "val_loss"]].mean()
        ax.plot(grouped["epoch"], grouped["val_loss"], marker="o", label=f"{model_name} val")
    ax.set_title("Validation Loss Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation BCE loss")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_lsw_deep_experiments(workspace_root: Path, lsw_root: Path) -> dict[str, object]:
    workspace_root = workspace_root.resolve()
    lsw_root = lsw_root.resolve()
    artifacts_root = lsw_root / "报告" / "实验产物"
    tables_dir = artifacts_root / "tables"
    figures_dir = artifacts_root / "figures"
    models_dir = artifacts_root / "models"
    for path in (tables_dir, figures_dir, models_dir):
        path.mkdir(parents=True, exist_ok=True)

    config = Step4Config.from_project_root(workspace_root)
    max_epochs = int(os.environ.get("LSW_DEEP_MAX_EPOCHS", "10"))
    n_splits = int(os.environ.get("LSW_DEEP_N_SPLITS", str(config.n_splits)))
    batch_size = int(os.environ.get("LSW_DEEP_BATCH_SIZE", "48"))
    config.max_epochs = max_epochs
    config.n_splits = n_splits
    config.batch_size = batch_size
    config.early_stopping_patience = int(os.environ.get("LSW_DEEP_PATIENCE", "3"))
    config.dropout = float(os.environ.get("LSW_DEEP_DROPOUT", "0.35"))

    _set_seed(config.random_seed)
    torch.backends.cudnn.benchmark = torch.cuda.is_available()
    spectrograms, labels, metadata_df, label_columns, _, dataset_info = load_step4_dataset(config)
    requested_device = os.environ.get("LSW_DEEP_DEVICE", "").strip().lower()
    if requested_device:
        device = torch.device(requested_device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splitter = KFold(n_splits=config.n_splits, shuffle=True, random_state=config.random_seed)
    model_names = ["cnn_baseline_lsw", "attention_bigru"]
    window_ids = metadata_df["window_id"].astype(str).tolist()

    fold_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    epoch_rows: list[dict[str, object]] = []
    oof_frames: list[pd.DataFrame] = [pd.DataFrame({"window_id": window_ids})]
    saved_models: list[str] = []
    best_model_name = ""
    best_lrap = -1.0
    labels_int = labels.astype(np.int64)

    for model_name in model_names:
        start = time.perf_counter()
        oof_scores = np.zeros_like(labels, dtype=np.float32)
        model_fold_rows: list[dict[str, object]] = []
        for fold_index, (train_idx, valid_idx) in enumerate(splitter.split(spectrograms), start=1):
            train_ds = LswSpectrogramDataset(
                spectrograms,
                labels,
                np.asarray(train_idx),
                spec_augment=model_name == "attention_bigru",
            )
            valid_ds = LswSpectrogramDataset(spectrograms, labels, np.asarray(valid_idx), spec_augment=False)
            train_loader = DataLoader(
                train_ds,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=device.type == "cuda",
            )
            valid_loader = DataLoader(
                valid_ds,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=device.type == "cuda",
            )
            model = build_lsw_deep_model(model_name, len(label_columns), config.dropout).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=_pos_weight(labels[train_idx]).to(device))
            optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
            scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
            best_state: dict[str, torch.Tensor] | None = None
            best_loss = float("inf")
            best_epoch = 0
            stale_epochs = 0
            fold_start = time.perf_counter()
            for epoch in range(1, config.max_epochs + 1):
                model.train()
                train_losses: list[float] = []
                for inputs, batch_labels in train_loader:
                    inputs = inputs.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
                    batch_labels = batch_labels.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                        logits = model(inputs)
                        loss = criterion(logits, batch_labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    train_losses.append(float(loss.item()))
                train_loss = float(np.mean(train_losses)) if train_losses else 0.0
                val_loss, _, _ = _evaluate(model, valid_loader, criterion, device)
                scheduler.step(val_loss)
                epoch_rows.append(
                    {
                        "model": model_name,
                        "fold": fold_index,
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    }
                )
                if val_loss < best_loss - 1e-4:
                    best_loss = val_loss
                    best_epoch = epoch
                    stale_epochs = 0
                    best_state = deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
                else:
                    stale_epochs += 1
                if stale_epochs >= config.early_stopping_patience:
                    break
                print(
                    f"[LSW deep {model_name} fold {fold_index} epoch {epoch}] "
                    f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} device={device}",
                    flush=True,
                )
            if best_state is None:
                raise RuntimeError(f"{model_name} fold {fold_index} did not produce a checkpoint.")
            model.load_state_dict(best_state)
            val_loss, scores, targets = _evaluate(model, valid_loader, criterion, device)
            oof_scores[valid_idx, :] = scores
            metrics = compute_lsw_metrics(targets.astype(np.int64), scores, threshold=0.5)
            row = {
                "model": model_name,
                "fold": fold_index,
                "fit_seconds": round(time.perf_counter() - fold_start, 3),
                "best_epoch": best_epoch,
                "best_val_loss": float(best_loss),
                "final_val_loss": float(val_loss),
                **metrics,
            }
            fold_rows.append(row)
            model_fold_rows.append(row)
            checkpoint_path = models_dir / f"{model_name}_fold{fold_index}.pt"
            _save_checkpoint(
                checkpoint_path,
                model=model.cpu(),
                model_name=model_name,
                fold=fold_index,
                best_epoch=best_epoch,
                label_columns=label_columns,
                config=config,
            )
            saved_models.append(str(checkpoint_path.relative_to(lsw_root)))
            model.to(device)
            print(
                f"[LSW deep {model_name} fold {fold_index}] "
                f"LRAP={metrics['lrap']:.4f} microF1={metrics['micro_f1']:.4f} top3={metrics['top3_hit_rate']:.4f}"
            )

        threshold = find_best_global_threshold(labels_int, oof_scores)
        threshold_rows.append({"model": model_name, **threshold})
        summary = {
            "model": model_name,
            "fit_seconds": round(time.perf_counter() - start, 3),
            "best_threshold": threshold["threshold"],
        }
        for metric in LSW_METRIC_COLUMNS:
            values = [float(row[metric]) for row in model_fold_rows]
            summary[metric] = float(np.mean(values)) if values else 0.0
            summary[f"{metric}_std"] = float(np.std(values)) if values else 0.0
        summary_rows.append(summary)
        oof_frames.append(pd.DataFrame(oof_scores, columns=[f"{model_name}__{label}" for label in label_columns]))
        if float(summary["lrap"]) > best_lrap:
            best_lrap = float(summary["lrap"])
            best_model_name = model_name

    deep_summary_df = pd.DataFrame(summary_rows).sort_values("lrap", ascending=False)
    existing_summary_path = tables_dir / "lsw_model_summary.csv"
    if existing_summary_path.exists():
        full_summary_df = pd.concat([pd.read_csv(existing_summary_path), deep_summary_df], ignore_index=True)
    else:
        full_summary_df = deep_summary_df
    full_summary_df = full_summary_df.sort_values("lrap", ascending=False)

    deep_summary_df.to_csv(tables_dir / "lsw_deep_summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(tables_dir / "lsw_deep_fold_results.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(tables_dir / "lsw_deep_threshold_optimization.csv", index=False)
    pd.DataFrame(epoch_rows).to_csv(tables_dir / "lsw_deep_epoch_history.csv", index=False)
    pd.concat(oof_frames, axis=1).to_csv(tables_dir / "lsw_deep_oof_predictions.csv", index=False)
    full_summary_df.to_csv(existing_summary_path, index=False)
    _plot_deep_summary(deep_summary_df, figures_dir / "lsw_deep_model_comparison.png")
    _plot_epoch_history(pd.DataFrame(epoch_rows), figures_dir / "lsw_deep_training_curves.png")

    manifest = {
        "owner": OWNER_NAME,
        "dataset": "BirdCLEF2026",
        "status": "deep_completed",
        "generated_at": _now_iso(),
        "stage": "lsw_deep",
        "device": str(device),
        "models": model_names,
        "best_deep_model": best_model_name,
        "sample_count": int(dataset_info["sample_count"]),
        "label_column_count": int(dataset_info["label_column_count"]),
        "saved_models": saved_models,
        "deep_training_defaults": {
            "max_epochs": config.max_epochs,
            "n_splits": config.n_splits,
            "batch_size": config.batch_size,
            "dropout": config.dropout,
            "optimization": "pos_weight + dropout + early stopping + attention_bigru SpecAugment",
        },
    }
    with (tables_dir / "lsw_deep_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    lsw_manifest_path = tables_dir / "lsw_experiment_manifest.json"
    current = {}
    if lsw_manifest_path.exists():
        with lsw_manifest_path.open("r", encoding="utf-8") as handle:
            current = json.load(handle)
    current.update(manifest)
    current["status"] = "completed"
    with lsw_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(current, handle, indent=2, ensure_ascii=False)
    return manifest
