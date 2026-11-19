"""Recorded-run chunk discovery shared by device replay sources."""

from __future__ import annotations

import re
from pathlib import Path


def _natural_chunk_key(path: Path, suffix: str, label_order=None):
    name = path.name
    match = re.match(r"^chunk_(\d{6})_", name)
    if match:
        return 0, int(match.group(1)), name
    match = re.match(rf"^chunk_(.*)_(\d{{6}})\.{re.escape(suffix)}$", name)
    if match:
        label, index = match.groups()
        order = (label_order or {}).get(label.casefold(), 10**9)
        return 1, order, int(index), name
    return 2, path.stat().st_mtime_ns, name


def chunk_files(chunks_dir: Path, suffix: str, label_order=None):
    chunks_dir = Path(chunks_dir)
    if not chunks_dir.exists():
        return []
    return sorted(
        chunks_dir.rglob(f"chunk_*.{suffix}"),
        key=lambda path: _natural_chunk_key(path, suffix, label_order),
    )
