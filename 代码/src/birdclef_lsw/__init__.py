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

OWNER_NAME = "BirdCLEF2026 \u8bfe\u7a0b\u5b9e\u9a8c"
PROJECT_SLUG = "lsw"

_MODULE_PATH = Path(__file__).resolve()
if _MODULE_PATH.parents[2].name == "\u4ee3\u7801":
    LSW_ROOT = _MODULE_PATH.parents[3]
    WORKSPACE_ROOT = LSW_ROOT.parent
elif _MODULE_PATH.parents[2].name == "lsw":
    LSW_ROOT = _MODULE_PATH.parents[2]
    WORKSPACE_ROOT = LSW_ROOT.parent
else:
    WORKSPACE_ROOT = _MODULE_PATH.parents[2]
    LSW_ROOT = WORKSPACE_ROOT / "lsw"

ARTIFACTS_ROOT = LSW_ROOT / "\u62a5\u544a" / "\u5b9e\u9a8c\u4ea7\u7269"
REPORT_ROOT = LSW_ROOT / "\u62a5\u544a" / "\u8bfe\u7a0b\u62a5\u544a"
