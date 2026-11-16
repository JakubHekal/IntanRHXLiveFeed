"""HDF5 implementation of LEECH's chunked raw-data sink."""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class Hdf5ChunkWriter:
    """Write ``(channels, samples)`` microvolt chunks to rotated HDF5 files.

    The public surface mirrors the subset of ``ChunkWriter`` used by the
    experiment runner, so the runner can select a sink without special cases.
    """

    def __init__(
        self,
        sample_rate: float,
        num_channels: int = 1,
        channel_names: Optional[List[str]] = None,
        chunk_max_sec: float = 300.0,
        buffer_bytes: int = 0,
        flush_interval_sec: float = 1.0,
    ) -> None:
        self.sample_rate = float(sample_rate)
        self.num_channels = int(num_channels)
        self.channel_names = list(channel_names or [f"CH{i + 1:03d}" for i in range(self.num_channels)])
        self.chunk_max_sec = float(chunk_max_sec)
        self.chunk_max_samples = max(1, int(self.sample_rate * self.chunk_max_sec))
        self.flush_interval_sec = float(flush_interval_sec)

        self.raw_chunks_dir: Optional[Path] = None
        self.filename_prefix = "chunk"
        self._chunk_index = 0
        self._chunk_samples_written = 0
        self._total_samples = 0
        self._chunk_start_sample = 0
        self._current_chunk_path: Optional[Path] = None
        self._raw_chunk_paths: List[Path] = []
        self._file: Any = None
        self._data: Any = None
        self._markers_sample_index: Any = None
        self._markers_id: Any = None
        self._markers_name: Any = None
        self._last_flush = 0.0

    @property
    def sample_index(self) -> int:
        return self._total_samples

    def start_session(self, project_name: str, project_path: Any = None) -> tuple:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(project_name)).strip("._") or "session"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = Path(project_path) if project_path is not None else Path.cwd()
        run_dir = root / f"{safe}_{stamp}"
        self.raw_chunks_dir = run_dir / "raw_chunks"
        self.raw_chunks_dir.mkdir(parents=True, exist_ok=True)
        self._chunk_index = 0
        self._total_samples = 0
        self._raw_chunk_paths = []
        self._close_file()
        self._open_new_chunk_locked()
        return run_dir, self.raw_chunks_dir

    def append_data(self, arr: np.ndarray, marker_info: Any = None, force_flush: bool = False) -> None:
        data = np.asarray(arr)
        if data.ndim != 2:
            raise ValueError(f"HDF5 writer expects (channels, samples), got {data.shape}")
        if data.shape[0] != self.num_channels:
            raise ValueError(
                f"HDF5 writer expects {self.num_channels} channels, got {data.shape[0]}"
            )
        if data.shape[1] == 0:
            return
        if self._file is None:
            self._open_new_chunk_locked()
        if self._chunk_samples_written >= self.chunk_max_samples:
            self._rotate_chunk_locked()
        if self._chunk_samples_written + data.shape[1] > self.chunk_max_samples and self._chunk_samples_written > 0:
            self._rotate_chunk_locked()

        samples = int(data.shape[1])
        old = int(self._data.shape[1])
        self._data.resize((self.num_channels, old + samples))
        self._data[:, old:old + samples] = data.astype(np.float32, copy=False)

        self._append_markers(marker_info)
        self._chunk_samples_written += samples
        self._total_samples += samples
        if force_flush or (time.monotonic() - self._last_flush) >= self.flush_interval_sec:
            self._flush_locked()

    def close(self) -> None:
        self._flush_locked()
        self._close_file()

    def get_paths(self) -> Dict[str, Any]:
        return {
            "raw_chunks_dir": str(self.raw_chunks_dir) if self.raw_chunks_dir else None,
            "current_chunk": str(self._current_chunk_path) if self._current_chunk_path else None,
            "chunks": [str(p) for p in self._raw_chunk_paths],
        }

    def _open_new_chunk_locked(self) -> None:
        import h5py

        if self.raw_chunks_dir is None:
            raise RuntimeError("HDF5 raw_chunks_dir is not set")
        self.raw_chunks_dir.mkdir(parents=True, exist_ok=True)
        self._current_chunk_path = (
            self.raw_chunks_dir / f"chunk_{self.filename_prefix}_{self._chunk_index:06d}.h5"
        )
        self._raw_chunk_paths.append(self._current_chunk_path)
        self._file = h5py.File(self._current_chunk_path, "w")
        self._file.attrs.update({
            "format": "LEECH_MCS_MEA_HDF5",
            "format_version": 1,
            "device": "MCS MEA2100",
            "device_type": "mcs_mea2100",
            "sample_rate_hz": self.sample_rate,
            "chunk_index": self._chunk_index,
            "chunk_start_sample": self._chunk_start_sample,
            "unit": "uV",
            "created_utc": datetime.now(timezone.utc).isoformat(),
        })
        chunk_samples = max(1, min(int(self.sample_rate), 20000))
        self._data = self._file.create_dataset(
            "data",
            shape=(self.num_channels, 0),
            maxshape=(self.num_channels, None),
            dtype="float32",
            chunks=(self.num_channels, chunk_samples),
            compression="gzip",
            compression_opts=1,
            shuffle=True,
        )
        self._data.attrs["unit"] = "uV"
        self._file.create_dataset(
            "channels",
            data=list(self.channel_names),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        self._file.create_group("markers")
        self._markers_sample_index = self._file.create_dataset(
            "markers/sample_index", shape=(0,), maxshape=(None,), dtype="int64"
        )
        self._markers_id = self._file.create_dataset(
            "markers/marker_id", shape=(0,), maxshape=(None,), dtype="int32"
        )
        self._markers_name = self._file.create_dataset(
            "markers/marker_name", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype(encoding="utf-8")
        )
        self._chunk_samples_written = 0
        self._chunk_start_sample = self._total_samples
        self._file.attrs["chunk_start_sample"] = self._chunk_start_sample
        self._last_flush = time.monotonic()

    def _rotate_chunk_locked(self) -> None:
        self._flush_locked()
        self._close_file()
        self._chunk_index += 1
        self._open_new_chunk_locked()

    def _append_markers(self, marker_info: Any) -> None:
        if not isinstance(marker_info, dict) or not marker_info:
            return
        indices = []
        ids = []
        names = []
        for sample_idx, info in marker_info.items():
            indices.append(self._chunk_start_sample + int(sample_idx))
            ids.append(int(info.get("id", 0)))
            names.append(str(info.get("name", "")))
        if not indices:
            return
        start = int(self._markers_sample_index.shape[0])
        end = start + len(indices)
        for dataset in (self._markers_sample_index, self._markers_id, self._markers_name):
            dataset.resize((end,))
        self._markers_sample_index[start:end] = np.asarray(indices, dtype=np.int64)
        self._markers_id[start:end] = np.asarray(ids, dtype=np.int32)
        self._markers_name[start:end] = names

    def _flush_locked(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._last_flush = time.monotonic()

    def _close_file(self) -> None:
        if self._file is not None:
            try:
                self._file.flush()
            except Exception:
                pass
            try:
                self._file.close()
            except Exception:
                pass
        self._file = None
        self._data = None
        self._markers_sample_index = None
        self._markers_id = None
        self._markers_name = None
