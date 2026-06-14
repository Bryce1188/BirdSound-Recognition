from __future__ import annotations

import os
import posixpath
from pathlib import Path

import paramiko


CODE_ROOT = Path(__file__).resolve().parents[1]
LSW_ROOT = CODE_ROOT.parent
LOCAL_ARTIFACTS_DIR = LSW_ROOT / "报告" / "实验产物"
LOCAL_REPORT_INPUT_DIR = LSW_ROOT / "报告" / "课程报告" / "remote_inputs"

REMOTE_HOST = os.environ.get("LSW_REMOTE_HOST", "")
REMOTE_PORT = int(os.environ.get("LSW_REMOTE_PORT", "22"))
REMOTE_USER = os.environ.get("LSW_REMOTE_USER", "")
REMOTE_PASSWORD = os.environ.get("LSW_REMOTE_PASSWORD", "")
REMOTE_ROOT = os.environ.get("LSW_REMOTE_ROOT", "/home/user/lsw")

REMOTE_PATHS = [
    ("artifacts/lsw/", LOCAL_ARTIFACTS_DIR),
    ("artifacts/lsw_deep_gpu_detached.log", LOCAL_REPORT_INPUT_DIR),
    ("artifacts/lsw_deep_gpu_detached.pid", LOCAL_REPORT_INPUT_DIR),
    ("artifacts/lsw_web.log", LOCAL_REPORT_INPUT_DIR),
    ("artifacts/lsw_web.pid", LOCAL_REPORT_INPUT_DIR),
    ("progress.md", LOCAL_REPORT_INPUT_DIR),
]


def _download_path(sftp: paramiko.SFTPClient, remote_path: str, local_root: Path, remote_root: str) -> None:
    try:
        stat_result = sftp.stat(remote_path)
    except FileNotFoundError:
        print(f"skip missing: {remote_path}")
        return

    remote_relative = Path(posixpath.relpath(remote_path, remote_root))

    if stat_result.st_mode & 0o40000:
        for entry in sftp.listdir_attr(remote_path):
            child_remote = posixpath.join(remote_path, entry.filename)
            child_relative = Path(posixpath.relpath(child_remote, remote_root))
            child_local = local_root / child_relative
            if entry.st_mode & 0o40000:
                child_local.mkdir(parents=True, exist_ok=True)
                _download_path(sftp, child_remote, local_root, remote_root)
            else:
                child_local.parent.mkdir(parents=True, exist_ok=True)
                print(f"download {child_remote} -> {child_local}")
                sftp.get(child_remote, str(child_local))
        return

    local_file = local_root / remote_relative
    local_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"download {remote_path} -> {local_file}")
    sftp.get(remote_path, str(local_file))


def main() -> int:
    missing = [
        name
        for name, value in {
            "LSW_REMOTE_HOST": REMOTE_HOST,
            "LSW_REMOTE_USER": REMOTE_USER,
            "LSW_REMOTE_PASSWORD": REMOTE_PASSWORD,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing remote connection environment variables: {', '.join(missing)}")

    LOCAL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_REPORT_INPUT_DIR.mkdir(parents=True, exist_ok=True)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        REMOTE_HOST,
        port=REMOTE_PORT,
        username=REMOTE_USER,
        password=REMOTE_PASSWORD,
        timeout=20,
    )
    try:
        sftp = client.open_sftp()
        for relative_path, local_root in REMOTE_PATHS:
            remote_path = posixpath.join(REMOTE_ROOT, relative_path)
            _download_path(sftp, remote_path, local_root, REMOTE_ROOT)
        sftp.close()
    finally:
        client.close()

    print("LSW remote results pulled successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
