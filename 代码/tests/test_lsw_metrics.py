from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

TEST_ROOT = Path(__file__).resolve().parents[1]
for path in [TEST_ROOT / "src", TEST_ROOT.parent / "src"]:
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from birdclef_lsw.metrics import compute_lsw_metrics, find_best_global_threshold


class LswMetricsTest(unittest.TestCase):
    def test_compute_lsw_metrics_includes_ranking_threshold_and_topk_fields(self) -> None:
        y_true = np.array(
            [
                [1, 0, 0],
                [0, 1, 1],
                [0, 0, 1],
            ],
            dtype=np.int64,
        )
        y_score = np.array(
            [
                [0.90, 0.20, 0.10],
                [0.10, 0.80, 0.70],
                [0.25, 0.40, 0.60],
            ],
            dtype=np.float64,
        )

        metrics = compute_lsw_metrics(y_true, y_score, threshold=0.5, top_k=(1, 3))

        expected_keys = {
            "lrap",
            "label_ranking_loss",
            "coverage_error",
            "hamming_loss",
            "micro_f1",
            "macro_f1",
            "top1_hit_rate",
            "top3_hit_rate",
            "prediction_density",
        }
        self.assertTrue(expected_keys.issubset(metrics))
        self.assertAlmostEqual(metrics["top1_hit_rate"], 1.0)
        self.assertAlmostEqual(metrics["top3_hit_rate"], 1.0)
        self.assertGreater(metrics["lrap"], 0.99)
        self.assertLess(metrics["label_ranking_loss"], 0.01)

    def test_find_best_global_threshold_prefers_threshold_with_highest_micro_f1(self) -> None:
        y_true = np.array(
            [
                [1, 0],
                [0, 1],
                [1, 0],
                [0, 0],
            ],
            dtype=np.int64,
        )
        y_score = np.array(
            [
                [0.70, 0.40],
                [0.30, 0.80],
                [0.60, 0.20],
                [0.45, 0.35],
            ],
            dtype=np.float64,
        )

        result = find_best_global_threshold(y_true, y_score, candidate_thresholds=[0.3, 0.5, 0.7])

        self.assertEqual(result["threshold"], 0.5)
        self.assertAlmostEqual(result["micro_f1"], 1.0)
        self.assertIn("hamming_loss", result)


if __name__ == "__main__":
    unittest.main()
