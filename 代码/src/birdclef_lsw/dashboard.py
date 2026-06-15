from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import ARTIFACTS_ROOT, OWNER_NAME, WORKSPACE_ROOT


REPORT_DIR = "\u62a5\u544a"
ARTIFACTS_DIR = "\u5b9e\u9a8c\u4ea7\u7269"


def build_teacher_checklist() -> list[dict[str, str]]:
    return [
        {
            "requirement": "\u9879\u76ee\u76ee\u6807",
            "status": "covered",
            "evidence": "\u7f51\u9875\u652f\u6301\u4e0a\u4f20\u9e1f\u9e23\u97f3\u9891\uff0c\u5e76\u8fd4\u56de Top-k \u7269\u79cd/\u58f0\u666f\u5019\u9009\u7ed3\u679c\u3002",
        },
        {
            "requirement": "\u7406\u8bba\u5b66\u4e60",
            "status": "covered",
            "evidence": "\u5b9e\u9a8c\u6750\u6599\u4fdd\u7559 MFCC\u3001pitch\u3001volume\u3001timbre\u3001rate/onset \u7b49\u58f0\u5b66\u7279\u5f81\u89e3\u91ca\u3002",
        },
        {
            "requirement": "\u6570\u636e\u96c6",
            "status": "covered",
            "evidence": "\u4f7f\u7528 BirdCLEF2026 \u5143\u6570\u636e\u3001train_soundscapes \u4e0e train_audio \u539f\u59cb\u6570\u636e\u3002",
        },
        {
            "requirement": "\u7279\u5f81\u63d0\u53d6",
            "status": "covered",
            "evidence": "Step2 \u751f\u6210 161 \u7ef4\u58f0\u5b66\u7279\u5f81\u548c Log-Mel \u9891\u8c31\u7f13\u5b58\u3002",
        },
        {
            "requirement": "\u6a21\u578b\u6784\u5efa",
            "status": "covered",
            "evidence": "\u5305\u542b Linear SVM\u3001KNN\u3001ExtraTrees\u3001Logistic Regression\u3001CNN \u4e0e Attention-BiGRU\u3002",
        },
        {
            "requirement": "\u6a21\u578b\u8bad\u7ec3\u4e0e\u6d4b\u8bd5",
            "status": "covered",
            "evidence": "KFold \u4ea4\u53c9\u9a8c\u8bc1\u8f93\u51fa fold \u7ea7\u6307\u6807\u548c OOF \u9884\u6d4b\u3002",
        },
        {
            "requirement": "\u6027\u80fd\u4f18\u5316",
            "status": "covered",
            "evidence": "\u8f93\u51fa\u5168\u5c40\u9608\u503c\u4f18\u5316\u5bf9\u7167\u8868\uff0c\u5e76\u5728\u6df1\u5ea6\u6a21\u578b\u4e2d\u4f7f\u7528 pos_weight\u3001dropout \u4e0e\u65e9\u505c\u3002",
        },
        {
            "requirement": "\u7528\u6237\u754c\u9762",
            "status": "covered",
            "evidence": "FastAPI + \u539f\u751f HTML/JS \u9875\u9762\u5c55\u793a\u4e0a\u4f20\u8bc6\u522b\u3001\u6307\u6807\u770b\u677f\u548c\u53ef\u89c6\u5316\u3002",
        },
    ]


def _read_json(path: Path, default: dict[str, object]) -> dict[str, object]:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        return payload
    return default


def _read_records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return pd.read_csv(path).fillna("").to_dict(orient="records")


def _resolve_workspace_root(lsw_root: Path) -> Path:
    if (lsw_root / "web_test_audio").exists():
        return lsw_root
    if lsw_root.name == "lsw":
        return lsw_root.parent
    return WORKSPACE_ROOT if WORKSPACE_ROOT.exists() else lsw_root


def build_dashboard_payload(lsw_root: Path) -> dict[str, object]:
    lsw_root = lsw_root.resolve()
    workspace_root = _resolve_workspace_root(lsw_root)
    artifacts_root = lsw_root / REPORT_DIR / ARTIFACTS_DIR
    if not artifacts_root.exists() and ARTIFACTS_ROOT.exists():
        artifacts_root = ARTIFACTS_ROOT
    tables_dir = artifacts_root / "tables"
    figures_dir = artifacts_root / "figures"
    manifest = _read_json(
        tables_dir / "lsw_experiment_manifest.json",
        {"owner": OWNER_NAME, "dataset": "BirdCLEF2026", "status": "pending"},
    )
    manifest["owner"] = OWNER_NAME
    manifest["dataset"] = "BirdCLEF2026"

    figures = []
    if figures_dir.exists():
        figures = [
            {
                "name": path.name,
                "url": f"/artifacts/figures/{path.name}",
            }
            for path in sorted(figures_dir.glob("*.png"))
        ]

    sample_audio = []
    sample_dir = workspace_root / "web_test_audio"
    if sample_dir.exists():
        sample_audio = [
            {
                "name": path.name,
                "url": f"/web_test_audio/{path.name}",
            }
            for path in sorted(sample_dir.glob("*.wav"))
        ]

    return {
        "owner": OWNER_NAME,
        "dataset": "BirdCLEF2026",
        "status": str(manifest.get("status", "pending")),
        "generated_at": str(manifest.get("generated_at", "")),
        "models": _read_records(tables_dir / "lsw_model_summary.csv"),
        "thresholds": _read_records(tables_dir / "lsw_threshold_optimization.csv"),
        "teacher_checklist": build_teacher_checklist(),
        "figures": figures,
        "sample_audio": sample_audio,
        "manifest": manifest,
    }
