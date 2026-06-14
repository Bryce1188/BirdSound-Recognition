from __future__ import annotations

import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
LSW_ROOT = CODE_ROOT.parent
WORKSPACE_ROOT = LSW_ROOT.parent
for path in [CODE_ROOT / "src", WORKSPACE_ROOT / "src"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from birdclef_lsw.deep import run_lsw_deep_experiments  # noqa: E402


def main() -> int:
    manifest = run_lsw_deep_experiments(WORKSPACE_ROOT, LSW_ROOT)
    print("LSW deep experiments completed.")
    print(f"Best deep model: {manifest['best_deep_model']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
