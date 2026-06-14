from __future__ import annotations

import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
LSW_ROOT = CODE_ROOT.parent
WORKSPACE_ROOT = LSW_ROOT.parent
for path in [CODE_ROOT / "src", WORKSPACE_ROOT / "src"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from birdclef_lsw.dashboard import build_dashboard_payload  # noqa: E402


def main() -> int:
    payload = build_dashboard_payload(LSW_ROOT)
    required = ["owner", "dataset", "models", "teacher_checklist", "figures"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise RuntimeError(f"Dashboard payload missing keys: {missing}")
    if len(payload["teacher_checklist"]) != 8:
        raise RuntimeError("Teacher checklist must contain 8 items.")
    print(json.dumps({key: payload[key] for key in ["owner", "dataset", "status"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
