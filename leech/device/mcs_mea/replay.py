"""HDF5 replay source for the MCS MEA2100 file layout."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..replay_paths import chunk_files


class McsHdf5Source:
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


# Backwards-compatible alias.
Hdf5Source = McsHdf5Source
