"""
ChunkWriter: Manages CSV chunk file I/O, rotation, and persistence.

Encapsulates all CSV writing, chunk rotation, and file buffering logic
to keep the experiment runner simple.
"""
import csv
import time
from PyQt5 import QtCore
import numpy as np


class ChunkWriter:
    """Handles writing multi-channel data to chunked CSV files with automatic rotation."""

    def __init__(self, sample_rate, num_channels=1, channel_names=None, chunk_max_sec=300, buffer_bytes=1024 * 1024, flush_interval_sec=1.0):
        self.sample_rate = float(sample_rate)
        self.num_channels = int(num_channels)
        self.channel_names = list(channel_names) if channel_names else None
        self.chunk_max_sec = float(chunk_max_sec)
        self.buffer_bytes = int(buffer_bytes)
        self.flush_interval_sec = float(flush_interval_sec)
        
        self.chunk_max_samples = max(1, int(round(self.sample_rate * self.chunk_max_sec)))
        self.csv_file_handle = None
        self.csv_writer = None
        self.csv_sample_counter = 0
        self._chunk_samples_written = 0
        self._chunk_index = 0
        self._raw_chunk_paths = []
        self._current_chunk_path = None
        self._chunks_since_flush = 0
        self._last_csv_flush_t = 0.0
        self.raw_chunks_dir = None
        self.filename_prefix = None
        self._io_mutex = QtCore.QMutex()

    def append_data(self, arr: np.ndarray, marker_info: dict = None, force_flush: bool = False):
        if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 1:
            return

        with QtCore.QMutexLocker(self._io_mutex):
            if self.csv_file_handle is None:
                return

            num_ch = int(arr.shape[0])
            n_samples = int(arr.shape[1])
            offset = 0
            has_marker = marker_info and len(marker_info) > 0
            inv_fs = 1.0 / self.sample_rate

            while offset < n_samples:
                if self.csv_file_handle is None:
                    self._open_new_chunk_locked()

                space = max(0, int(self.chunk_max_samples - self._chunk_samples_written))
                if space <= 0:
                    self._rotate_chunk_locked()
                    continue

                take = min(space, n_samples - offset)
                seg_start = int(self.csv_sample_counter + offset)

                seg = arr[:, offset:offset + take].astype(np.float64, copy=False)
                times = (seg_start + np.arange(take, dtype=np.float64)) * inv_fs

                marker_ids = np.zeros(take, dtype=np.int64) if has_marker else None
                marker_names = [""] * take
                if has_marker:
                    for sample_idx, marker_rec in marker_info.items():
                        if sample_idx < seg_start or sample_idx >= seg_start + take:
                            continue
                        local = sample_idx - seg_start
                        marker_ids[local] = int(marker_rec.get("id", 0))
                        marker_names[local] = str(marker_rec.get("name", ""))
                rows = []
                for i in range(take):
                    ch_vals = [f"{seg[ch, i]:.4f}" for ch in range(num_ch)]
                    marker_id = int(marker_ids[i]) if has_marker else 0
                    rows.append([f"{times[i]:.6f}"] + ch_vals + [marker_id, marker_names[i]])
                self.csv_writer.writerows(rows)

                self._chunk_samples_written += take
                offset += take

                if self._chunk_samples_written >= self.chunk_max_samples:
                    self._rotate_chunk_locked()

            self._chunks_since_flush += 1
            now = time.perf_counter()
            if has_marker or force_flush or (now - self._last_csv_flush_t) >= self.flush_interval_sec:
                self.csv_file_handle.flush()
                self._chunks_since_flush = 0
                self._last_csv_flush_t = now

            self.csv_sample_counter += n_samples
    
    def _open_new_chunk_locked(self):
        if self.raw_chunks_dir is None:
            return
        self._chunk_index += 1
        tag = f"_{self.filename_prefix}" if self.filename_prefix else ""
        path = self.raw_chunks_dir / f"chunk{tag}_{self._chunk_index:06d}.csv"
        self.csv_file_handle = open(
            path,
            "w",
            newline="",
            encoding="utf-8",
            buffering=self.buffer_bytes,
        )
        self.csv_writer = csv.writer(self.csv_file_handle)
        if self.channel_names and len(self.channel_names) == self.num_channels:
            channel_columns = [f"{n}_uV" for n in self.channel_names]
        else:
            channel_columns = [f"ch_{i}_uV" for i in range(self.num_channels)]
        header = ["time_s"] + channel_columns + ["marker_id", "marker_name"]
        self.csv_writer.writerow(header)
        self._chunk_samples_written = 0
        self._current_chunk_path = path
        self._raw_chunk_paths.append(path)
    
    def _rotate_chunk_locked(self):
        """Close current chunk and open a new one (must hold _io_mutex)."""
        if self.csv_file_handle is not None:
            try:
                self.csv_file_handle.flush()
                self.csv_file_handle.close()
            except Exception:
                pass
        self.csv_file_handle = None
        self.csv_writer = None
        self._current_chunk_path = None
        self._chunk_samples_written = 0
        self._open_new_chunk_locked()
        self._last_csv_flush_t = time.perf_counter()
    
    def close(self):
        """Close current chunk and finalize writer."""
        with QtCore.QMutexLocker(self._io_mutex):
            if self.csv_file_handle is not None:
                try:
                    self.csv_file_handle.flush()
                    self.csv_file_handle.close()
                except Exception:
                    pass
            self.csv_file_handle = None
            self.csv_writer = None
            self._current_chunk_path = None
