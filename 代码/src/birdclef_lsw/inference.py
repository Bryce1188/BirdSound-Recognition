from __future__ import annotations

import os
import uuid
from pathlib import Path

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from . import ARTIFACTS_ROOT, WORKSPACE_ROOT
from .deep import build_lsw_deep_model


def _resolve_workspace_root(lsw_root: Path) -> Path:
    if (lsw_root / "artifacts" / "step2").exists():
        return lsw_root
    if lsw_root.name == "lsw":
        return lsw_root.parent
    return WORKSPACE_ROOT if WORKSPACE_ROOT.exists() else lsw_root


def _find_checkpoint(lsw_root: Path) -> Path:
    models_dir = lsw_root / "报告" / "实验产物" / "models"
    if not models_dir.exists():
        models_dir = ARTIFACTS_ROOT / "models"
    preferred = [
        models_dir / "attention_bigru_fold1.pt",
        models_dir / "cnn_baseline_lsw_fold1.pt",
    ]
    for path in preferred:
        if path.exists():
            return path
    candidates = sorted(models_dir.glob("*.pt")) if models_dir.exists() else []
    if candidates:
        return candidates[0]
    raise FileNotFoundError("No LSW deep checkpoint found under lsw/报告/实验产物/models.")


def _normalise_audio(audio: np.ndarray, expected_length: int | None = None) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if expected_length is not None:
        if audio.size > expected_length:
            audio = audio[:expected_length]
        if audio.size < expected_length:
            audio = np.pad(audio, (0, expected_length - audio.size))
    if audio.size == 0:
        audio = np.zeros(expected_length or 32000, dtype=np.float32)
    audio = audio - float(np.mean(audio))
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.0:
        audio = audio / peak
    return audio.astype(np.float32, copy=False)


def _segment_audio(audio: np.ndarray, sample_rate: int, window_seconds: float) -> np.ndarray:
    expected = int(round(sample_rate * window_seconds))
    if audio.size == 0:
        audio = np.zeros(expected, dtype=np.float32)
    segment_count = max(1, int(np.ceil(audio.size / expected)))
    padded = np.pad(audio, (0, segment_count * expected - audio.size))
    return padded.reshape(segment_count, expected).astype(np.float32, copy=False)


def _log_mel(
    audio: np.ndarray,
    *,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_mels: int,
    f_min: int,
    f_max: int,
    top_db: int,
) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max,
        power=2.0,
        center=True,
    )
    return librosa.power_to_db(mel, ref=np.max, top_db=top_db).astype(np.float32, copy=False)


def _load_active_label_lookup(workspace_root: Path) -> dict[str, dict[str, str]]:
    path = workspace_root / "artifacts" / "step2" / "tables" / "active_labels.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path).fillna("")
    return {str(row["label"]): row for row in df.to_dict(orient="records")}


def _save_waveform(audio: np.ndarray, sample_rate: int, output_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(9, 3))
    times = np.arange(audio.size) / float(sample_rate)
    ax.plot(times, audio, color="#31572c", linewidth=0.7)
    ax.set_title("Uploaded Audio Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    name = f"waveform_{uuid.uuid4().hex}.png"
    fig.savefig(output_dir / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return f"/artifacts/web_cache/{name}"


def _save_matrix(matrix: np.ndarray, output_dir: Path, *, title: str, prefix: str, cmap: str) -> str:
    fig, ax = plt.subplots(figsize=(9, 4))
    image = ax.imshow(matrix, origin="lower", aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("Frames")
    ax.set_ylabel("Coefficients / Bands")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    name = f"{prefix}_{uuid.uuid4().hex}.png"
    fig.savefig(output_dir / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return f"/artifacts/web_cache/{name}"


def predict_audio_file(lsw_root: Path, audio_path: Path) -> dict[str, object]:
    lsw_root = lsw_root.resolve()
    workspace_root = _resolve_workspace_root(lsw_root)
    checkpoint_path = _find_checkpoint(lsw_root)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    label_columns = [str(label) for label in checkpoint["label_columns"]]
    sample_rate = int(config.get("sample_rate", 32000))
    window_seconds = float(config.get("duration_seconds", 5.0))
    model_name = str(checkpoint.get("model_name") or config.get("architecture", "attention_bigru"))
    if model_name not in {"cnn_baseline_lsw", "attention_bigru"}:
        model_name = "attention_bigru"

    model = build_lsw_deep_model(model_name, len(label_columns), float(config.get("dropout", 0.35)))
    model.load_state_dict(checkpoint["state_dict"])
    requested_device = os.environ.get("LSW_INFER_DEVICE", "cpu").strip().lower()
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    model.to(device)
    model.eval()

    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    audio = _normalise_audio(audio)
    segments = _segment_audio(audio, sample_rate, window_seconds)
    specs = np.stack(
        [
            _log_mel(
                segment,
                sample_rate=sample_rate,
                n_fft=int(config.get("n_fft", 2048)),
                hop_length=int(config.get("hop_length", 512)),
                win_length=int(config.get("win_length", 2048)),
                n_mels=int(config.get("n_mels", 128)),
                f_min=int(config.get("f_min", 50)),
                f_max=int(config.get("f_max", 14000)),
                top_db=int(config.get("top_db", 80)),
            )
            for segment in segments
        ]
    )
    normalized = np.clip((specs + 80.0) / 80.0, 0.0, 1.0).astype(np.float32)
    inputs = torch.from_numpy(normalized[:, None, :, :]).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        logits = model(inputs)
        probabilities = torch.sigmoid(logits).detach().cpu().numpy()
    aggregated = probabilities.max(axis=0)
    top_indices = np.argsort(-aggregated)[:5]
    lookup = _load_active_label_lookup(workspace_root)
    top_rows = []
    for index in top_indices:
        label = label_columns[int(index)]
        meta = lookup.get(label, {})
        top_rows.append(
            {
                "label": label,
                "score": round(float(aggregated[int(index)]), 6),
                "common_name": str(meta.get("common_name", "")),
                "scientific_name": str(meta.get("scientific_name", "")),
                "class_name": str(meta.get("class_name", "")),
            }
        )

    cache_dir = lsw_root / "报告" / "实验产物" / "web_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    mfcc = librosa.feature.mfcc(y=audio[: int(sample_rate * min(15.0, max(window_seconds, 1.0)))], sr=sample_rate, n_mfcc=20)
    return {
        "model": model_name,
        "checkpoint": str(checkpoint_path.relative_to(lsw_root)),
        "duration_seconds": round(float(audio.size) / float(sample_rate), 3),
        "sample_rate": sample_rate,
        "window_count": int(segments.shape[0]),
        "top_predictions": top_rows,
        "figures": {
            "waveform": _save_waveform(audio, sample_rate, cache_dir),
            "log_mel": _save_matrix(specs[0], cache_dir, title="Log-Mel Spectrogram", prefix="logmel", cmap="magma"),
            "mfcc": _save_matrix(mfcc, cache_dir, title="MFCC Features", prefix="mfcc", cmap="coolwarm"),
        },
    }
