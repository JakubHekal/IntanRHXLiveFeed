"""CSV replay source for the ChunkWriter layout.

Device packages with a different CSV schema should provide their own source.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .replay_paths import chunk_files

_CSV_EXCLUDED = {"time_s", "marker_id", "marker_name"}


class CsvSource:
    """Sequential block reader for old CSV runs; never parses a whole file."""

    def __init__(self, chunks_dir: Path, sample_rate: float = 20000.0, label_order=None):
        self._files = chunk_files(chunks_dir, "csv", label_order)
        if not self._files:
            raise FileNotFoundError(f"No CSV chunks in {chunks_dir}")
        self._file_idx = -1
        self._handle = None
        self._columns = ()
        self._ncols = 0
        self._labels = []
        self._sample_rate = float(sample_rate or 20000.0)
        self._samples_done = 0
        self._exhausted = False
        self._pending_lines = []
        self._load_header(0)

    # ── metadata ────────────────────────────────────────────────────────
    def _load_header(self, index: int):
        self._close_handle()
        self._file_idx = index
        self._handle = self._files[index].open("r", encoding="utf-8-sig", newline="")
        header_line = self._handle.readline()
        header = [part.strip() for part in header_line.rstrip("\r\n").split(",")]
        self._columns = tuple(
            i for i, name in enumerate(header) if name.lower() not in _CSV_EXCLUDED
        )
        self._ncols = len(header)
        if not self._labels:
            self._labels = [header[i].replace("_uV", "") or f"CH{i}" for i in self._columns]

    def _close_handle(self):
        if self._handle is not None:
            try:
                self._handle.close()
            finally:
                self._handle = None

    def close(self):
        self._close_handle()
        self._exhausted = True

    @property
    def sample_rate(self) -> float:
        return self._sample_rate

    @property
    def num_channels(self) -> int:
        return len(self._columns)

    @property
    def labels(self):
        return list(self._labels)

    @property
    def total_samples(self):
        return None

    @property
    def exhausted(self) -> bool:
        return self._exhausted

    # ── reading ─────────────────────────────────────────────────────────
    def _take_lines(self, n: int):
        lines = []
        while len(lines) < n:
            line = self._handle.readline()
            if not line:
                if self._file_idx + 1 < len(self._files):
                    self._load_header(self._file_idx + 1)
                    continue
                self._exhausted = True
                break
            if line.strip():
                lines.append(line.rstrip("\r\n"))
        return lines

    def read(self, n: int):
        if self._exhausted or n <= 0:
            return None, []
        lines = self._take_lines(int(n))
        if not lines:
            return None, []
        # Drop the free-text marker_name before numeric parsing.
        numeric_cols = self._ncols - 1
        numeric_lines = [",".join(line.split(",")[:numeric_cols]) for line in lines]
        flat = np.fromstring(",".join(numeric_lines), sep=",", dtype=np.float32)
        if flat.size != len(lines) * numeric_cols:
            rows = max(1, flat.size // max(1, numeric_cols))
            flat = flat[: rows * numeric_cols]
            lines = lines[:rows]
        table = flat.reshape(len(lines), numeric_cols)
        chunk = table[:, self._columns].T.copy()

        markers = []
        # marker columns are the last two in every writer output
        if self._ncols >= 3:
            marker_ids = table[:, numeric_cols - 1]
            for row, marker_id in enumerate(marker_ids):
                if marker_id == 0:
                    continue
                name = lines[row].split(",")[-1].strip()
                markers.append((row, int(marker_id), name or f"Marker {int(marker_id)}"))
        self._samples_done += chunk.shape[1]
        return chunk, markers
