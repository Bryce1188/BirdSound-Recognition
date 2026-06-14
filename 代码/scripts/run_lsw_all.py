from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
LSW_ROOT = CODE_ROOT.parent
WORKSPACE_ROOT = LSW_ROOT.parent


def _run(script_name: str) -> None:
    script_root = WORKSPACE_ROOT / "scripts" if script_name.startswith("run_step") else CODE_ROOT / "scripts"
    command = [sys.executable, str(script_root / script_name)]
    print(f"\n>>> {' '.join(command)}", flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(CODE_ROOT / "src"), str(WORKSPACE_ROOT / "src")])
    result = subprocess.run(command, cwd=str(WORKSPACE_ROOT), env=env, text=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> int:
    for script in [
        "run_step1_eda.py",
        "run_step2_features.py",
        "run_step3_ml.py",
        "run_step4_mel_cache.py",
        "run_lsw_classical_experiments.py",
        "run_lsw_deep_experiments.py",
    ]:
        _run(script)
    print("\nLSW full experiment pipeline completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
