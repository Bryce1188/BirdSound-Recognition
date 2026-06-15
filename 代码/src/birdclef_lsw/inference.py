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


DATASET_DIR = "\u6570\u636e\u96c6"
REPORT_DIR = "\u62a5\u544a"
ARTIFACTS_DIR = "\u5b9e\u9a8c\u4ea7\u7269"


def _resolve_workspace_root(lsw_root: Path) -> Path:
    if (lsw_root / "artifacts" / "step2").exists():
        return lsw_root
    if lsw_root.name == "lsw":
        return lsw_root.parent
    return WORKSPACE_ROOT if WORKSPACE_ROOT.exists() else lsw_root


def _find_checkpoint(lsw_root: Path) -> Path:
    models_dir = lsw_root / REPORT_DIR / ARTIFACTS_DIR / "models"
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
    raise FileNotFoundError(f"No LSW deep checkpoint found under lsw/{REPORT_DIR}/{ARTIFACTS_DIR}/models.")


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
    candidates = [
        workspace_root / "artifacts" / "step2" / "tables" / "active_labels.csv",
        workspace_root / "birdclef-2026" / "taxonomy.csv",
        workspace_root / "birdclef-2026" / "train.csv",
        workspace_root / DATASET_DIR / "taxonomy.csv",
        workspace_root / DATASET_DIR / "train.csv",
        Path("D:/birdclef-2026/taxonomy.csv"),
        Path("D:/birdclef-2026/train.csv"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        df = pd.read_csv(path).fillna("")
        if "label" in df.columns:
            key_column = "label"
        elif "primary_label" in df.columns:
            key_column = "primary_label"
        elif "inat_taxon_id" in df.columns:
            key_column = "inat_taxon_id"
        else:
            continue
        records: dict[str, dict[str, str]] = {}
        for row in df.to_dict(orient="records"):
            key = str(row.get(key_column, "")).strip()
            if not key or key in records:
                continue
            records[key] = {str(k): str(v) for k, v in row.items()}
        if records:
            return records
    return {}


_CLASS_DISPLAY_NAMES = {
    "Aves": "\u9e1f\u7c7b",
    "Amphibia": "\u4e24\u6816\u7c7b",
    "Insecta": "\u6606\u866b\u7c7b",
    "Mammalia": "\u54fa\u4e73\u7c7b",
}


_CHINESE_NAME_BY_LABEL = {
    "bafcur1": "\u88f8\u8138\u51a0\u96c9",
    "bufpar": "\u84dd\u989d\u4e9a\u9a6c\u900a\u9e66\u9e49",
    "bunibi1": "\u9ec4\u9888\u9e6e",
    "chacha1": "\u67e5\u79d1\u5c0f\u51a0\u96c9",
    "chvcon1": "\u6817\u81c0\u9525\u5634\u96c0",
    "compau": "\u666e\u901a\u591c\u9e70",
    "compot1": "\u666e\u901a\u6797\u9e31",
    "fusfly1": "\u6697\u8272\u9738\u9e5f",
    "grekis": "\u5927\u98df\u8747\u9738\u9e5f",
    "hyamac1": "\u7d2b\u84dd\u91d1\u521a\u9e66\u9e49",
    "limpki": "\u79e7\u9e64",
    "litnig1": "\u5c0f\u591c\u9e70",
    "magant1": "\u9a6c\u6258\u683c\u7f57\u7d22\u8681\u9e1f",
    "nacnig1": "\u7eb3\u6606\u8fbe\u591c\u9e70",
    "orwpar": "\u6a59\u7fc5\u4e9a\u9a6c\u900a\u9e66\u9e49",
    "plcjay1": "\u7ed2\u51a0\u84dd\u9e26",
    "purjay1": "\u7d2b\u84dd\u9e26",
    "redjun": "\u7ea2\u539f\u9e21",
    "rufhor2": "\u68d5\u7076\u9e1f",
    "ruther1": "\u6817\u8272\u864e\u9e6d",
    "rutjac1": "\u68d5\u5c3e\u7fe0\u9d97",
    "sibtan2": "\u94f6\u5634\u5510\u7eb3\u96c0",
    "strher2": "\u7eff\u9e6d",
    "thlwre1": "\u9e2b\u5f62\u9e6a\u9e69",
    "trsowl": "\u70ed\u5e26\u9e23\u89d2\u9e2e",
    "undtin1": "\u6ce2\u7eb9\u9e44\u9e1f",
    "wfwduc1": "\u767d\u8138\u6811\u9e2d",
    "whtdov": "\u767d\u5c16\u9e20",
}


def _species_payload(label: str, meta: dict[str, str]) -> dict[str, str]:
    common_name = str(meta.get("common_name", "")).strip()
    scientific_name = str(meta.get("scientific_name", "")).strip()
    class_name = str(meta.get("class_name", "")).strip()
    class_display_name = _CLASS_DISPLAY_NAMES.get(class_name, class_name)
    species_code = str(meta.get("primary_label") or meta.get("label") or meta.get("inat_taxon_id") or label).strip()
    chinese_name = _CHINESE_NAME_BY_LABEL.get(species_code, "")
    species_name = common_name or scientific_name or species_code or label
    if chinese_name and species_name:
        display_name = f"{chinese_name} ({species_name})"
    else:
        display_name = species_name if species_name != label else f"\u7269\u79cd\u6807\u7b7e {label}"
    return {
        "display_name": display_name,
        "chinese_name": chinese_name,
        "species_name": species_name,
        "species_code": species_code or label,
        "common_name": common_name,
        "scientific_name": scientific_name,
        "class_name": class_name,
        "class_display_name": class_display_name,
    }


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
    lookup = _load_active_label_lookup(workspace_root)
    ranked_indices = np.argsort(-aggregated)
    ranked_rows = []
    for index in ranked_indices:
        raw_label = label_columns[int(index)]
        meta = lookup.get(raw_label, {})
        species = _species_payload(raw_label, meta)
        ranked_rows.append(
            {
                "label": species["display_name"],
                "label_id": raw_label,
                **species,
                "score": round(float(aggregated[int(index)]), 6),
            }
        )
    top_soundscape_rows = ranked_rows[:5]
    top_bird_rows = [row for row in ranked_rows if row.get("class_name") == "Aves"][:5]
    top_rows = top_bird_rows or top_soundscape_rows

    cache_dir = lsw_root / REPORT_DIR / ARTIFACTS_DIR / "web_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    mfcc = librosa.feature.mfcc(
        y=audio[: int(sample_rate * min(15.0, max(window_seconds, 1.0)))],
        sr=sample_rate,
        n_mfcc=20,
    )
    return {
        "model": model_name,
        "checkpoint": str(checkpoint_path.relative_to(lsw_root)),
        "duration_seconds": round(float(audio.size) / float(sample_rate), 3),
        "sample_rate": sample_rate,
        "window_count": int(segments.shape[0]),
        "prediction_scope": "birds" if top_bird_rows else "soundscape",
        "top_predictions": top_rows,
        "top_bird_predictions": top_bird_rows,
        "top_soundscape_predictions": top_soundscape_rows,
        "figures": {
            "waveform": _save_waveform(audio, sample_rate, cache_dir),
            "log_mel": _save_matrix(specs[0], cache_dir, title="Log-Mel Spectrogram", prefix="logmel", cmap="magma"),
            "mfcc": _save_matrix(mfcc, cache_dir, title="MFCC Features", prefix="mfcc", cmap="coolwarm"),
        },
    }
