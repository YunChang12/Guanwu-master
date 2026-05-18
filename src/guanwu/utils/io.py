"""I/O utilities."""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import orjson


def read_json(path: str | Path) -> dict | list:
    """Read a JSON file using orjson."""
    with open(path, "rb") as f:
        return orjson.loads(f.read())


def write_json(path: str | Path, data: dict | list) -> None:
    """Write a JSON file using orjson."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))


def file_sha256(path: str | Path) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def safe_copy(src: Path, dst: Path, overwrite: bool = False) -> Path:
    """Copy a file, refusing to overwrite unless told to."""
    if dst.exists() and not overwrite:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def ensure_dir(path: str | Path) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
