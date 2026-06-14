from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

TEST_ROOT = Path(__file__).resolve().parents[1]
for path in [TEST_ROOT / "src", TEST_ROOT.parent / "src"]:
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from birdclef_lsw.dashboard import build_dashboard_payload, build_teacher_checklist


class LswDashboardTest(unittest.TestCase):
    def test_build_dashboard_payload_uses_fixed_public_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tables = root / "artifacts" / "lsw" / "tables"
            figures = root / "artifacts" / "lsw" / "figures"
            tables.mkdir(parents=True)
            figures.mkdir(parents=True)

            pd.DataFrame(
                [
                    {
                        "model": "extra_trees",
                        "lrap": 0.82,
                        "micro_f1": 0.44,
                        "top3_hit_rate": 0.91,
                        "hamming_loss": 0.08,
                    }
                ]
            ).to_csv(tables / "lsw_model_summary.csv", index=False)
            (tables / "lsw_experiment_manifest.json").write_text(
                json.dumps({"owner": "李少威", "dataset": "BirdCLEF2026"}, ensure_ascii=False),
                encoding="utf-8",
            )

            payload = build_dashboard_payload(root)

        self.assertEqual(payload["owner"], "BirdCLEF2026 课程实验")
        self.assertEqual(payload["dataset"], "BirdCLEF2026")
        self.assertEqual(payload["models"][0]["model"], "extra_trees")
        self.assertIn("teacher_checklist", payload)
        self.assertIn("figures", payload)

    def test_teacher_checklist_covers_required_eight_items(self) -> None:
        checklist = build_teacher_checklist()

        self.assertEqual(len(checklist), 8)
        self.assertTrue(all(item["status"] == "covered" for item in checklist))
        self.assertEqual(checklist[0]["requirement"], "项目目标")


if __name__ == "__main__":
    unittest.main()
