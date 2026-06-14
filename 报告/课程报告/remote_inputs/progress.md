# BirdCLEF 2026 Progress Board

## Overview

| Module | Status | Completion | Last Updated | Owner | Notes |
|---|---|---:|---|---|---|
| Step 1 Data Intake and EDA | Completed | 100% | 2026-06-11T16:29:46+08:00 | Codex | Local dataset used, no Kaggle CLI |
| Step 2 Feature Engineering | Completed | 100% | 2026-06-11T16:36:02+08:00 | Codex | Speech-oriented 5-second window features; GPU reserved for later DL stages |
| Step 3 Classical ML | Completed | 100% | 2026-06-11T16:37:48+08:00 | Codex | Classical ML baselines on speech-style acoustic features |
| Step 4 CNN Baseline | Completed | 100% | 2026-04-10T15:39:49+08:00 | Codex | Lightweight CNN on log-Mel spectrograms with GPU training and Step 3 comparison |
| Step 5 UI | Completed | 100% | 2026-05-29T11:09:45+08:00 | Codex | Gradio UI for CNN inference plus speech-recognition feature visualization |
| Step 6 Report Packaging | Not Started | 0% | - | - | Pending |

## Current Tasks

| ID | Task | Status | Output | Started | Finished |
|---|---|---|---|---|---|
| S3-01 | Load and align Step 2 features with label matrix | Completed | Aligned X/Y matrices keyed by window_id | 2026-06-11T16:36:05+08:00 | 2026-06-11T16:36:05+08:00 |
| S3-02 | Train 5-fold classical ML baselines | Completed | Logistic regression, linear SVM, random forest, KNN | 2026-06-11T16:36:05+08:00 | 2026-06-11T16:37:48+08:00 |
| S3-03 | Export CV tables, OOF predictions, manifests, and plots | Completed | CSV/JSON summaries and report-ready figures under artifacts/step3 | 2026-06-11T16:37:48+08:00 | 2026-06-11T16:37:48+08:00 |
| S3-04 | Refresh dynamic progress board | Completed | progress.md and shared project progress state | 2026-06-11T16:36:04+08:00 | 2026-06-11T16:37:48+08:00 |

## Latest Run

- Module: `Step 3 Classical ML`
- Status: `completed`
- Environment: `bin`
- Started: `2026-06-11T16:36:04+08:00`
- Finished: `2026-06-11T16:37:48+08:00`
- Key Metrics:
  - `sample_count`: `739`
  - `feature_column_count`: `161`
  - `label_column_count`: `75`
  - `model_count`: `4`
  - `best_macro_roc_auc`: `0.9721661681850335`
  - `gpu_count`: `1`
  - `torch_cuda_available`: `True`
  - `progress_board`: `progress.md`

## Logs

| Time | Module | Action | Result |
|---|---|---|---|
| 2026-06-11T16:36:04+08:00 | Step 3 | Initialize pipeline | Using environment bin |
| 2026-06-11T16:36:05+08:00 | Step 3 | Check runtime environment | gpu_count=1, torch_cuda_available=True |
| 2026-06-11T16:36:05+08:00 | Step 3 | Load dataset | samples=739, feature_columns=161, label_columns=75, duplicates_removed=739 |
| 2026-06-11T16:37:48+08:00 | Step 3 | Run cross-validation | models=4, folds=5, oof_rows=739 |
| 2026-06-11T16:37:48+08:00 | Step 3 | Write artifacts | tables=6, figures=2 |

## Artifacts

| Type | Path |
|---|---|
| Table | `artifacts/step3/tables/cv_results.csv` |
| Table | `artifacts/step3/tables/fold_results.csv` |
| Table | `artifacts/step3/tables/per_label_results.csv` |
| Table | `artifacts/step3/tables/oof_predictions.csv` |
| Table | `artifacts/step3/tables/training_manifest.json` |
| Table | `artifacts/step3/tables/training_qc.json` |
| Figure | `artifacts/step3/figures/model_comparison.png` |
| Figure | `artifacts/step3/figures/per_label_auc_top_bottom.png` |

## Next Steps

- Keep the project centered on speech-oriented audio preprocessing and interpretable acoustic features.
- Compare classical machine learning and lightweight deep learning under the same 5-second window protocol.
- Preserve `progress.md` as the single status board for all later stages.
