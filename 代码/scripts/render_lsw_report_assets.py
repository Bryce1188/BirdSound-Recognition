from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

CODE_ROOT = Path(__file__).resolve().parents[1]
LSW_ROOT = CODE_ROOT.parent
WORKSPACE_ROOT = LSW_ROOT.parent
for path in [CODE_ROOT / "src", WORKSPACE_ROOT / "src"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from birdclef_step3 import Step3Config
from birdclef_step3.data import load_step3_dataset

ARTIFACTS_DIR = LSW_ROOT / "报告" / "实验产物"
TABLES_DIR = ARTIFACTS_DIR / "tables"
FIGURES_SRC_DIR = ARTIFACTS_DIR / "figures"
SCREENSHOTS_SRC_DIR = ARTIFACTS_DIR / "screenshots"
WEB_CACHE_DIR = ARTIFACTS_DIR / "web_cache"

REPORT_ROOT = LSW_ROOT / "报告" / "课程报告"
FIGURES_DIR = REPORT_ROOT / "figures"
GENERATED_DIR = REPORT_ROOT / "generated"

NMI = {
    "ink": "#303047",
    "muted": "#70708A",
    "axis": "#484878",
    "grid": "#E5E6F0",
    "blue": "#7884B4",
    "blue_light": "#B4C0E4",
    "teal": "#A5B6D8",
    "teal_light": "#E4E4F0",
    "green": "#E4CCD8",
    "warm": "#C98298",
    "warm_light": "#F0C0CC",
    "purple": "#8C78A8",
    "gray": "#D8D8D8",
}

MODEL_LABELS = {
    "extra_trees": "ExtraTrees",
    "logistic_regression_l1": "LogReg-L1",
    "knn_distance": "KNN",
    "linear_svm": "Linear SVM",
    "cnn_baseline_lsw": "CNN baseline",
    "attention_bigru": "Attention-BiGRU",
}

MODEL_COLORS = {
    "extra_trees": NMI["purple"],
    "logistic_regression_l1": NMI["warm_light"],
    "knn_distance": NMI["blue_light"],
    "linear_svm": NMI["blue"],
    "cnn_baseline_lsw": NMI["teal_light"],
    "attention_bigru": NMI["green"],
}


def _apply_nmi_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "Source Han Sans SC",
                "Arial",
                "DejaVu Sans",
                "sans-serif",
            ],
            "axes.unicode_minus": False,
            "mathtext.fontset": "dejavusans",
            "font.size": 8.6,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 8.1,
            "ytick.labelsize": 8.1,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.65,
            "axes.edgecolor": NMI["axis"],
            "axes.labelcolor": NMI["ink"],
            "xtick.color": NMI["ink"],
            "ytick.color": NMI["ink"],
            "lines.linewidth": 1.25,
            "patch.linewidth": 0.55,
            "savefig.dpi": 420,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def _clean_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(NMI["axis"])
        spine.set_linewidth(0.65)
    ax.tick_params(axis="both", width=0.65, length=2.8, pad=2.2, colors=NMI["ink"])
    ax.grid(axis=grid_axis, color=NMI["grid"], linewidth=0.65, alpha=0.88, zorder=0)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.05,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=NMI["ink"],
    )


def _display_model(name: object) -> str:
    return MODEL_LABELS.get(str(name), str(name))


def _model_color(name: object) -> str:
    return MODEL_COLORS.get(str(name), NMI["blue"])


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_path, dpi=420, bbox_inches="tight")
    plt.close(fig)


def _copy_if_exists(source: Path, target: Path) -> None:
    if not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _escape_tex(text: object) -> str:
    value = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def _write_tabular(df: pd.DataFrame, output_path: Path, columns: list[tuple[str, str, str]]) -> None:
    spec = "".join(colspec for _, _, colspec in columns)
    lines = [rf"\begin{{tabular}}{{{spec}}}", r"\toprule"]
    headers = " & ".join(_escape_tex(title) for _, title, _ in columns) + r" \\"
    lines.append(headers)
    lines.append(r"\midrule")
    for row in df.to_dict(orient="records"):
        rendered = []
        for key, _, _ in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                rendered.append(f"{value:.4f}")
            else:
                rendered.append(_escape_tex(value))
        lines.append(" & ".join(rendered) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _copy_latest_visual(prefix: str, target_name: str) -> None:
    candidates = sorted(WEB_CACHE_DIR.glob(f"{prefix}_*.png"), key=lambda path: path.stat().st_mtime, reverse=True)
    if candidates:
        _copy_if_exists(candidates[0], FIGURES_DIR / target_name)


def _feature_family_counts(feature_columns: list[str]) -> pd.DataFrame:
    counts = {
        "base": 0,
        "mfcc": 0,
        "pitch": 0,
        "volume": 0,
        "timbre": 0,
        "rhythm_rate": 0,
    }
    for name in feature_columns:
        if name in {"duration_sec", "sample_rate"}:
            counts["base"] += 1
        elif name.startswith(("mfcc_", "delta_mfcc_", "delta2_mfcc_")):
            counts["mfcc"] += 1
        elif name.startswith("f0_") or name == "voiced_ratio":
            counts["pitch"] += 1
        elif name.startswith("rms_") or name == "dynamic_range":
            counts["volume"] += 1
        elif name.startswith(("spectral_", "zero_crossing_")):
            counts["timbre"] += 1
        elif name.startswith(("onset_", "peak_")):
            counts["rhythm_rate"] += 1
        else:
            raise ValueError(f"Unclassified feature column: {name}")
    rows = [
        {"family": "基础统计", "count": counts["base"], "description": "duration_sec, sample_rate"},
        {"family": "MFCC及差分", "count": counts["mfcc"], "description": "20个MFCC + 一阶/二阶差分统计"},
        {"family": "音高Pitch", "count": counts["pitch"], "description": "f0_mean/std/min/max + voiced_ratio"},
        {"family": "音量Volume", "count": counts["volume"], "description": "rms_mean/std/max + dynamic_range"},
        {"family": "音色Timbre", "count": counts["timbre"], "description": "谱质心/带宽/滚降/平坦度/谱对比度/ZCR"},
        {"family": "节奏Rate", "count": counts["rhythm_rate"], "description": "onset_count/rate/strength + peak_count/rate"},
    ]
    return pd.DataFrame(rows)


def _dataset_statistics() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = Step3Config.from_project_root(WORKSPACE_ROOT)
    _, labels_df, feature_columns, label_columns, _, _, _ = load_step3_dataset(config)
    label_counts = labels_df.sum(axis=1)
    per_label = labels_df.sum(axis=0).sort_values(ascending=False)

    stats_df = pd.DataFrame(
        [
            {"item": "唯一5秒窗口数", "value": int(labels_df.shape[0])},
            {"item": "激活标签数", "value": int(len(label_columns))},
            {"item": "结构化特征维度", "value": int(len(feature_columns))},
            {"item": "窗口平均标签数", "value": round(float(label_counts.mean()), 4)},
            {"item": "窗口标签数中位数", "value": round(float(label_counts.median()), 4)},
            {"item": "多标签窗口比例", "value": round(float((label_counts > 1).mean()), 4)},
        ]
    )
    feature_family_df = _feature_family_counts(feature_columns)
    top_labels_df = per_label.head(10).reset_index()
    top_labels_df.columns = ["label", "positive_windows"]
    return stats_df, feature_family_df, top_labels_df


def _plot_top_labels(top_labels_df: pd.DataFrame, output_path: Path) -> None:
    _apply_nmi_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    colors = [NMI["purple"]] + [NMI["blue"] if i < 5 else NMI["blue_light"] for i in range(len(top_labels_df) - 1)]
    ax.bar(top_labels_df["label"].astype(str), top_labels_df["positive_windows"], color=colors, edgecolor=NMI["axis"])
    ax.set_title("Top-10 active labels by positive windows", loc="left", color=NMI["ink"], fontweight="bold")
    ax.set_ylabel("Positive windows")
    _clean_axis(ax)
    for tick in ax.get_xticklabels():
        tick.set_rotation(25)
        tick.set_ha("right")
    _save_figure(fig, output_path)


def _plot_label_density(labels_df: pd.DataFrame, output_path: Path) -> None:
    _apply_nmi_style()
    counts = labels_df.sum(axis=1)
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    bins = range(1, int(counts.max()) + 2)
    ax.hist(counts, bins=bins, align="left", rwidth=0.86, color=NMI["blue_light"], edgecolor=NMI["axis"])
    ax.axvline(float(counts.mean()), color=NMI["warm"], linestyle="--", linewidth=1.2, label=f"Mean={counts.mean():.2f}")
    ax.set_title("Active labels per 5-second window", loc="left", color=NMI["ink"], fontweight="bold")
    ax.set_xlabel("Number of active labels")
    ax.set_ylabel("Window count")
    _clean_axis(ax)
    ax.legend(frameon=False, loc="upper right")
    _save_figure(fig, output_path)


def _plot_overall_summary(summary_df: pd.DataFrame, output_path: Path) -> None:
    _apply_nmi_style()
    ordered = summary_df.sort_values("lrap", ascending=False).copy()
    ordered["display"] = ordered["model"].map(_display_model)
    y = np.arange(len(ordered))
    fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.9), sharey=True)
    specs = [
        ("lrap", "LRAP", "lrap_std", True),
        ("micro_f1", "Micro F1", "micro_f1_std", True),
        ("top3_hit_rate", "Top-3 hit", "top3_hit_rate_std", True),
        ("hamming_loss", "Hamming loss", "hamming_loss_std", False),
    ]
    for idx, (ax, (metric, title, std_col, higher_better)) in enumerate(zip(axes, specs)):
        colors = [_model_color(model) for model in ordered["model"]]
        ax.barh(y, ordered[metric], xerr=ordered.get(std_col), color=colors, edgecolor=NMI["axis"], alpha=0.92)
        ax.set_title(title, color=NMI["ink"], fontweight="bold")
        ax.invert_yaxis()
        if idx == 0:
            ax.set_yticks(y, ordered["display"])
        else:
            ax.tick_params(labelleft=False)
        if higher_better:
            ax.set_xlim(0.0, 1.0)
        else:
            ax.set_xlim(0.0, max(0.7, float(ordered[metric].max()) * 1.12))
        _clean_axis(ax, grid_axis="x")
        _panel_label(ax, chr(ord("A") + idx))
    fig.suptitle("Model-level comparison under multi-label ranking metrics", x=0.01, ha="left", color=NMI["ink"], fontweight="bold")
    _save_figure(fig, output_path)


def _plot_classical_model_comparison(summary_df: pd.DataFrame, output_path: Path) -> None:
    _apply_nmi_style()
    df = summary_df[summary_df["model"].isin(["extra_trees", "logistic_regression_l1", "knn_distance", "linear_svm"])].copy()
    df = df.sort_values("lrap", ascending=True)
    y = np.arange(len(df))
    fig, axes = plt.subplots(1, 3, figsize=(9.7, 3.6), sharey=True)
    specs = [
        ("lrap", "LRAP", "lrap_std", True),
        ("micro_f1", "Micro F1", "micro_f1_std", True),
        ("hamming_loss", "Hamming loss", "hamming_loss_std", False),
    ]
    for idx, (ax, (metric, title, std_col, higher_better)) in enumerate(zip(axes, specs)):
        ax.errorbar(
            df[metric],
            y,
            xerr=df[std_col],
            fmt="o",
            color=NMI["axis"],
            ecolor=NMI["axis"],
            elinewidth=1.1,
            capsize=2.4,
            markersize=6.2,
            markerfacecolor=NMI["purple"] if idx == 0 else NMI["blue"],
            markeredgecolor="white",
            zorder=3,
        )
        ax.set_title(title, color=NMI["ink"], fontweight="bold")
        if idx == 0:
            ax.set_yticks(y, df["model"].map(_display_model))
        else:
            ax.tick_params(labelleft=False)
        if higher_better:
            ax.set_xlim(max(0.70, float(df[metric].min()) - 0.05), min(1.0, float(df[metric].max()) + 0.05))
        else:
            ax.set_xlim(0.0, max(0.045, float(df[metric].max()) * 1.25))
        xmin, xmax = ax.get_xlim()
        offset = (xmax - xmin) * 0.035
        for yi, value in zip(y, df[metric]):
            label_x = min(value + offset, xmax - offset)
            ax.text(
                label_x,
                yi,
                f"{value:.3f}",
                ha="left",
                va="center",
                color=NMI["muted"],
                fontsize=7.1,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.25},
            )
        _clean_axis(ax, grid_axis="x")
        _panel_label(ax, chr(ord("A") + idx))
    fig.suptitle("Traditional models: stable acoustic features favor ExtraTrees", x=0.01, ha="left", color=NMI["ink"], fontweight="bold")
    _save_figure(fig, output_path)


def _plot_threshold_nmi(threshold_df: pd.DataFrame, output_path: Path) -> None:
    _apply_nmi_style()
    df = threshold_df.copy().sort_values("lrap", ascending=False)
    df["display"] = df["model"].map(_display_model)
    x = np.arange(len(df))
    fig, axes = plt.subplots(1, 2, figsize=(8.9, 3.4))
    axes[0].bar(x, df["threshold"], color=[_model_color(model) for model in df["model"]], edgecolor=NMI["axis"])
    axes[0].set_xticks(x, df["display"], rotation=20, ha="right")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Optimized threshold")
    axes[0].set_title("Threshold search", color=NMI["ink"], fontweight="bold")
    _clean_axis(axes[0])
    _panel_label(axes[0], "A")

    axes[1].plot(df["display"], df["micro_f1"], marker="o", color=NMI["purple"], label="Micro F1")
    axes[1].plot(df["display"], df["prediction_density"], marker="s", color=NMI["warm"], label="Prediction density")
    axes[1].set_xticks(x, df["display"], rotation=20, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("F1-density trade-off", color=NMI["ink"], fontweight="bold")
    _clean_axis(axes[1])
    axes[1].legend(frameon=False, loc="upper right")
    _panel_label(axes[1], "B")
    fig.suptitle("Global threshold tuning aligns predictions with multi-label density", x=0.01, ha="left", color=NMI["ink"], fontweight="bold")
    _save_figure(fig, output_path)


def _plot_deep_summary_nmi(summary_df: pd.DataFrame, output_path: Path) -> None:
    _apply_nmi_style()
    df = summary_df.copy().sort_values("lrap", ascending=False)
    x = np.arange(len(df))
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.2))
    specs = [("lrap", "LRAP"), ("top3_hit_rate", "Top-3 hit"), ("micro_f1", "Micro F1")]
    for idx, (ax, (metric, title)) in enumerate(zip(axes, specs)):
        ax.bar(x, df[metric], color=[_model_color(model) for model in df["model"]], edgecolor=NMI["axis"])
        ax.set_xticks(x, df["model"].map(_display_model), rotation=16, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title(title, color=NMI["ink"], fontweight="bold")
        _clean_axis(ax)
        _panel_label(ax, chr(ord("A") + idx))
    fig.suptitle("Deep models remain a contrastive spectral baseline", x=0.01, ha="left", color=NMI["ink"], fontweight="bold")
    _save_figure(fig, output_path)


def _plot_epoch_history_nmi(history_df: pd.DataFrame, output_path: Path) -> None:
    _apply_nmi_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for model_name, model_df in history_df.groupby("model"):
        grouped = model_df.groupby("epoch", as_index=False)[["train_loss", "val_loss"]].mean()
        ax.plot(
            grouped["epoch"],
            grouped["val_loss"],
            marker="o",
            label=f"{_display_model(model_name)} val",
            color=_model_color(model_name),
        )
    ax.set_title("Validation loss shows unstable spectral learning", loc="left", color=NMI["ink"], fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation BCE loss")
    _clean_axis(ax)
    ax.legend(frameon=False, loc="upper left")
    _save_figure(fig, output_path)


def _plot_rank_heatmap(summary_df: pd.DataFrame, output_path: Path) -> None:
    _apply_nmi_style()
    metrics = ["lrap", "micro_f1", "top3_hit_rate", "hamming_loss"]
    titles = ["LRAP", "Micro F1", "Top-3", "Hamming"]
    ranks = []
    ordered_models = summary_df.sort_values("lrap", ascending=False)["model"].tolist()
    for model in ordered_models:
        row = []
        model_row = summary_df[summary_df["model"] == model].iloc[0]
        for metric in metrics:
            ascending = metric == "hamming_loss"
            rank = summary_df[metric].rank(method="min", ascending=ascending).loc[summary_df["model"] == model].iloc[0]
            row.append(rank)
        ranks.append(row)
    data = np.array(ranks, dtype=float)
    cmap = LinearSegmentedColormap.from_list("nmi_rank", [NMI["blue_light"], "#FFFFFF", NMI["warm_light"]])
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    im = ax.imshow(data, cmap=cmap, vmin=1, vmax=float(len(summary_df)))
    ax.set_xticks(np.arange(len(metrics)), titles)
    ax.set_yticks(np.arange(len(ordered_models)), [_display_model(model) for model in ordered_models])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{int(data[i, j])}", ha="center", va="center", color=NMI["ink"], fontsize=9)
    ax.set_title("Metric rank heatmap (1 is best)", loc="left", color=NMI["ink"], fontweight="bold")
    for spine in ax.spines.values():
        spine.set_color(NMI["axis"])
        spine.set_linewidth(0.65)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Rank")
    cbar.outline.set_edgecolor(NMI["axis"])
    _save_figure(fig, output_path)


def main() -> int:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(TABLES_DIR / "lsw_model_summary.csv").sort_values("lrap", ascending=False)
    deep_summary_df = pd.read_csv(TABLES_DIR / "lsw_deep_summary.csv").sort_values("lrap", ascending=False)
    classical_threshold_df = pd.read_csv(TABLES_DIR / "lsw_threshold_optimization.csv").sort_values("lrap", ascending=False)
    deep_threshold_df = pd.read_csv(TABLES_DIR / "lsw_deep_threshold_optimization.csv").sort_values("lrap", ascending=False)
    deep_fold_df = pd.read_csv(TABLES_DIR / "lsw_deep_fold_results.csv").sort_values(["model", "fold"])
    deep_epoch_df = pd.read_csv(TABLES_DIR / "lsw_deep_epoch_history.csv")
    manifest = json.loads((TABLES_DIR / "lsw_experiment_manifest.json").read_text(encoding="utf-8"))
    deep_manifest = json.loads((TABLES_DIR / "lsw_deep_manifest.json").read_text(encoding="utf-8"))

    stats_df, feature_family_df, top_labels_df = _dataset_statistics()

    # Generated copies for traceability.
    summary_df.to_csv(GENERATED_DIR / "lsw_model_summary_copy.csv", index=False)
    deep_summary_df.to_csv(GENERATED_DIR / "lsw_deep_summary_copy.csv", index=False)
    classical_threshold_df.to_csv(GENERATED_DIR / "lsw_threshold_optimization_copy.csv", index=False)
    deep_threshold_df.to_csv(GENERATED_DIR / "lsw_deep_threshold_optimization_copy.csv", index=False)
    deep_fold_df.to_csv(GENERATED_DIR / "lsw_deep_fold_results_copy.csv", index=False)
    deep_epoch_df.to_csv(GENERATED_DIR / "lsw_deep_epoch_history_copy.csv", index=False)
    stats_df.to_csv(GENERATED_DIR / "lsw_dataset_statistics.csv", index=False)
    feature_family_df.to_csv(GENERATED_DIR / "lsw_feature_families.csv", index=False)
    top_labels_df.to_csv(GENERATED_DIR / "lsw_top_labels.csv", index=False)
    (GENERATED_DIR / "lsw_manifest_snapshot.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (GENERATED_DIR / "lsw_deep_manifest_snapshot.json").write_text(
        json.dumps(deep_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # TeX tables.
    _write_tabular(
        stats_df,
        GENERATED_DIR / "lsw_dataset_statistics_table.tex",
        [("item", "统计项", "p{5.5cm}"), ("value", "数值", "r")],
    )
    _write_tabular(
        feature_family_df,
        GENERATED_DIR / "lsw_feature_families_table.tex",
        [("family", "特征家族", "p{3.0cm}"), ("count", "维度", "r"), ("description", "说明", "p{7.8cm}")],
    )
    _write_tabular(
        summary_df[["model", "lrap", "micro_f1", "macro_f1", "top3_hit_rate", "hamming_loss", "best_threshold"]],
        GENERATED_DIR / "lsw_model_summary_table.tex",
        [
            ("model", "Model", "l"),
            ("lrap", "LRAP", "r"),
            ("micro_f1", "Micro F1", "r"),
            ("macro_f1", "Macro F1", "r"),
            ("top3_hit_rate", "Top-3", "r"),
            ("hamming_loss", "Hamming", "r"),
            ("best_threshold", "Thr", "r"),
        ],
    )
    _write_tabular(
        deep_fold_df[["model", "fold", "best_epoch", "lrap", "micro_f1", "top3_hit_rate"]],
        GENERATED_DIR / "lsw_deep_fold_table.tex",
        [
            ("model", "Model", "l"),
            ("fold", "Fold", "r"),
            ("best_epoch", "Best epoch", "r"),
            ("lrap", "LRAP", "r"),
            ("micro_f1", "Micro F1", "r"),
            ("top3_hit_rate", "Top-3", "r"),
        ],
    )
    combined_threshold_df = pd.concat([classical_threshold_df, deep_threshold_df], ignore_index=True)
    _write_tabular(
        combined_threshold_df[["model", "threshold", "lrap", "micro_f1", "prediction_density"]],
        GENERATED_DIR / "lsw_threshold_table.tex",
        [
            ("model", "Model", "l"),
            ("threshold", "Threshold", "r"),
            ("lrap", "LRAP", "r"),
            ("micro_f1", "Micro F1", "r"),
            ("prediction_density", "Pred density", "r"),
        ],
    )

    # Figures generated from current final results.
    _plot_overall_summary(summary_df, FIGURES_DIR / "lsw_model_overview.png")
    _plot_top_labels(top_labels_df, FIGURES_DIR / "lsw_top_labels.png")

    config = Step3Config.from_project_root(WORKSPACE_ROOT)
    _, labels_df, _, _, _, _, _ = load_step3_dataset(config)
    _plot_label_density(labels_df, FIGURES_DIR / "lsw_label_density.png")

    # Copy existing experiment figures and screenshots.
    _copy_if_exists(FIGURES_SRC_DIR / "lsw_classical_model_comparison.png", FIGURES_DIR / "lsw_classical_model_comparison.png")
    _copy_if_exists(FIGURES_SRC_DIR / "lsw_deep_model_comparison.png", FIGURES_DIR / "lsw_deep_model_comparison.png")
    _copy_if_exists(FIGURES_SRC_DIR / "lsw_deep_training_curves.png", FIGURES_DIR / "lsw_deep_training_curves.png")
    _copy_if_exists(FIGURES_SRC_DIR / "lsw_threshold_optimization.png", FIGURES_DIR / "lsw_threshold_optimization.png")

    # Report-only polished figures. These intentionally overwrite the copied
    # experiment PNGs inside the report folder without changing artifacts/web.
    _plot_classical_model_comparison(summary_df, FIGURES_DIR / "lsw_classical_model_comparison.png")
    _plot_threshold_nmi(classical_threshold_df, FIGURES_DIR / "lsw_threshold_optimization.png")
    _plot_deep_summary_nmi(deep_summary_df, FIGURES_DIR / "lsw_deep_model_comparison.png")
    _plot_epoch_history_nmi(deep_epoch_df, FIGURES_DIR / "lsw_deep_training_curves.png")
    _plot_rank_heatmap(summary_df, FIGURES_DIR / "lsw_metric_rank_heatmap.png")

    _copy_if_exists(SCREENSHOTS_SRC_DIR / "lsw_dashboard_home.png", FIGURES_DIR / "lsw_dashboard_home.png")
    _copy_if_exists(SCREENSHOTS_SRC_DIR / "lsw_dashboard_demo.png", FIGURES_DIR / "lsw_dashboard_demo.png")
    _copy_if_exists(SCREENSHOTS_SRC_DIR / "lsw_dashboard_audio_module.png", FIGURES_DIR / "lsw_dashboard_audio_module.png")
    _copy_if_exists(SCREENSHOTS_SRC_DIR / "lsw_dashboard_metrics_module.png", FIGURES_DIR / "lsw_dashboard_metrics_module.png")
    _copy_if_exists(SCREENSHOTS_SRC_DIR / "lsw_dashboard_checklist_module.png", FIGURES_DIR / "lsw_dashboard_checklist_module.png")
    _copy_if_exists(SCREENSHOTS_SRC_DIR / "lsw_dashboard_figures_module.png", FIGURES_DIR / "lsw_dashboard_figures_module.png")
    _copy_latest_visual("waveform", "lsw_ui_waveform.png")
    _copy_latest_visual("logmel", "lsw_ui_logmel.png")
    _copy_latest_visual("mfcc", "lsw_ui_mfcc.png")

    print("LSW report assets rendered.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
