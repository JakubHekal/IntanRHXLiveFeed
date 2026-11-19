"""Streaming replay sources for recorded runs.

Both sources expose the same small interface used by ReplaySession:

    source.sample_rate        -> float
    source.num_channels       -> int
    source.labels             -> list[str]
    source.total_samples      -> int | None   (None when not known up front)
    source.read(n)            -> (chunk, markers) | (None, [])
    source.exhausted          -> bool

``chunk`` is a float32 matrix of shape (channels, samples) in microvolts.
``markers`` is a list of (offset_in_chunk, marker_id, marker_name).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_CSV_EXCLUDED = {"time_s", "marker_id", "marker_name"}


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


class Hdf5Source:
    """Slice reader for the HDF5 layout written by Hdf5ChunkWriter."""

    _STREAM = "Data/Recording_0/AnalogStream/Stream_0"

    def __init__(self, chunks_dir: Path, label_order=None):
        import h5py  # local import keeps startup light for non-hdf5 runs

        self._h5py = h5py
        self._files = chunk_files(chunks_dir, "h5", label_order)
        if not self._files:
            raise FileNotFoundError(f"No HDF5 chunks in {chunks_dir}")
        self._sizes = []
        self._offsets = []
        self._handle = None
        self._handle_idx = -1
        self._marker_cache = {}
        self._labels = []
        self._sample_rate = 20000.0
        self._conversion = None
        self._load_metadata()
        total = 0
        for size in self._sizes:
            self._offsets.append(total)
            total += size
        self._total = total
        self._cursor = 0

    def _open(self, index: int):
        if self._handle_idx == index and self._handle is not None:
            return self._handle
        self._close()
        self._handle = self._h5py.File(self._files[index], "r")
        self._handle_idx = index
        return self._handle

    def _close(self):
        if self._handle is not None:
            try:
                self._handle.close()
            finally:
                self._handle = None
                self._handle_idx = -1

    def close(self):
        self._close()

    def _load_metadata(self):
        import h5py

        for path in self._files:
            with h5py.File(path, "r") as handle:
                self._sizes.append(int(handle[f"{self._STREAM}/ChannelData"].shape[1]))
        first = self._files[0]
        with h5py.File(first, "r") as handle:
            info = handle[f"{self._STREAM}/InfoChannel"][:]
            labels = []
            ticks = []
            conversions = []
            for row in info:
                label = row["Label"]
                if isinstance(label, bytes):
                    label = label.decode("utf-8", "replace")
                labels.append(str(label))
                ticks.append(float(row["Tick"]))
                conversions.append(float(row["ConversionFactor"]))
            self._labels = labels
            tick = ticks[0] if ticks else 50.0
            self._sample_rate = 1e6 / tick if tick > 0 else 20000.0
            self._conversion = np.asarray(conversions, dtype=np.float64) * 1e-6

    @property
    def sample_rate(self) -> float:
        return self._sample_rate

    @property
    def num_channels(self) -> int:
        return len(self._labels)

    @property
    def labels(self):
        return list(self._labels)

    @property
    def total_samples(self):
        return self._total

    @property
    def exhausted(self) -> bool:
        return self._cursor >= self._total

    def _markers_for(self, index: int):
        if index in self._marker_cache:
            return self._marker_cache[index]
        handle = self._open(index)
        markers = []
        try:
            group = handle["markers"]
            indices = group["sample_index"][:]
            ids = group["marker_id"][:] if "marker_id" in group else np.zeros_like(indices)
            names = group["marker_name"][:]
            for sample, marker_id, name in zip(indices, ids, names):
                if isinstance(name, bytes):
                    name = name.decode("utf-8", "replace")
                markers.append((int(sample), int(marker_id), str(name)))
        except (KeyError, ValueError):
            markers = []
        self._marker_cache[index] = markers
        return markers

    def read(self, n: int):
        if self._cursor >= self._total or n <= 0:
            return None, []
        n = min(int(n), self._total - self._cursor)
        start = self._cursor
        end = start + n
        pieces = []
        markers = []
        for index, (offset, size) in enumerate(zip(self._offsets, self._sizes)):
            file_end = offset + size
            if file_end <= start or offset >= end:
                continue
            handle = self._open(index)
            local_start = max(0, start - offset)
            local_end = min(size, end - offset)
            raw = handle[f"{self._STREAM}/ChannelData"][:, local_start:local_end]
            scale = self._conversion[:, np.newaxis]
            pieces.append((raw.astype(np.float32) * scale).astype(np.float32, copy=False))
            for sample, marker_id, name in self._markers_for(index):
                if offset + sample < end and offset + sample >= start:
                    markers.append((offset + sample - start, marker_id, name))
        self._cursor = end
        if not pieces:
            return None, []
        chunk = pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=1)
        return chunk, markers


def make_source(chunks_dir: Path, fmt: str, sample_rate: float = 20000.0, label_order=None):
    if fmt == "hdf5":
        return Hdf5Source(chunks_dir, label_order=label_order)
    if fmt == "csv":
        return CsvSource(chunks_dir, sample_rate=sample_rate, label_order=label_order)
    raise ValueError(f"Unsupported replay format: {fmt}")
