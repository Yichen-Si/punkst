"""Shared process and request-file helpers for multires orchestration."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any


def write_request(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, path.stat().st_mode & 0o777
                 if path.exists() else 0o644)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def run_native(punkst: str, command: str, request: Path,
               output: Path) -> dict[str, Any]:
    executable = shutil.which(punkst) if os.sep not in punkst else punkst
    if executable is None:
        raise FileNotFoundError(f"Cannot find punkst executable: {punkst}")
    completed = subprocess.run(
        [str(executable), command, "--request", str(request),
         "--out-dir", str(output)],
        check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"punkst {command} failed: {detail}")
    try:
        status = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"punkst {command} returned invalid status JSON") from error
    if not isinstance(status, dict):
        raise RuntimeError(f"punkst {command} returned invalid status")
    return status
