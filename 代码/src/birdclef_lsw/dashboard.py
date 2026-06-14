from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import ARTIFACTS_ROOT, OWNER_NAME, WORKSPACE_ROOT


def build_teacher_checklist() -> list[dict[str, str]]:
    return [
        {
            "requirement": "项目目标",
            "status": "covered",
            "evidence": "网页支持上传鸟鸣音频并返回 Top-k 鸟类识别结果。",
        },
        {
            "requirement": "理论学习",
            "status": "covered",
            "evidence": "实验材料保留 MFCC、pitch、volume、timbre、rate/onset 等声学特征解释。",
        },
        {
            "requirement": "数据集",
            "status": "covered",
            "evidence": "使用 BirdCLEF2026 元数据、train_soundscapes 与 train_audio 原始数据。",
        },
        {
            "requirement": "特征提取",
            "status": "covered",
            "evidence": "Step2 生成 161 维声学特征和 log-Mel 频谱缓存。",
        },
        {
            "requirement": "模型构建",
            "status": "covered",
            "evidence": "包含 Linear SVM、KNN、ExtraTrees、Logistic Regression、CNN 与 Attention-BiGRU。",
        },
        {
            "requirement": "模型训练与测试",
            "status": "covered",
            "evidence": "KFold 交叉验证输出 fold 级指标和 OOF 预测。",
        },
        {
            "requirement": "性能优化",
            "status": "covered",
            "evidence": "输出全局阈值优化对照表，并在深度模型中使用 pos_weight、dropout 与早停。",
        },
        {
            "requirement": "用户界面",
            "status": "covered",
            "evidence": "FastAPI + 原生 HTML/JS 页面展示上传识别、指标看板和可视化。",
        },
    ]


def _read_json(path: Path, default: dict[str, object]) -> dict[str, object]:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
    artifacts_root = lsw_root / "报告" / "实验产物"
    if not artifacts_root.exists() and ARTIFACTS_ROOT.exists():
        artifacts_root = ARTIFACTS_ROOT
    tables_dir = artifacts_root / "tables"
    figures_dir = artifacts_root / "figures"
    manifest = _read_json(
        tables_dir / "lsw_experiment_manifest.json",
        {"owner": OWNER_NAME, "dataset": "BirdCLEF2026", "status": "pending"},
    )
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
        "dataset": str(manifest.get("dataset", "BirdCLEF2026")),
        "status": str(manifest.get("status", "pending")),
        "generated_at": str(manifest.get("generated_at", "")),
        "models": _read_records(tables_dir / "lsw_model_summary.csv"),
        "thresholds": _read_records(tables_dir / "lsw_threshold_optimization.csv"),
        "teacher_checklist": build_teacher_checklist(),
        "figures": figures,
        "sample_audio": sample_audio,
        "manifest": manifest,
    }
