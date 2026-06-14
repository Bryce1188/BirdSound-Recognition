from __future__ import annotations

from pathlib import Path

__all__ = [
    "ARTIFACTS_ROOT",
    "LSW_ROOT",
    "OWNER_NAME",
    "PROJECT_SLUG",
    "REPORT_ROOT",
    "WORKSPACE_ROOT",
]

OWNER_NAME = "BirdCLEF2026 课程实验"
PROJECT_SLUG = "lsw"

_MODULE_PATH = Path(__file__).resolve()
if _MODULE_PATH.parents[2].name == "代码":
    LSW_ROOT = _MODULE_PATH.parents[3]
    WORKSPACE_ROOT = LSW_ROOT.parent
elif _MODULE_PATH.parents[2].name == "lsw":
    LSW_ROOT = _MODULE_PATH.parents[2]
    WORKSPACE_ROOT = LSW_ROOT.parent
else:
    WORKSPACE_ROOT = _MODULE_PATH.parents[2]
    LSW_ROOT = WORKSPACE_ROOT / "lsw"

ARTIFACTS_ROOT = LSW_ROOT / "报告" / "实验产物"
REPORT_ROOT = LSW_ROOT / "报告" / "课程报告"
