"""
ChunkWriter: Manages CSV chunk file I/O, rotation, and persistence.

Encapsulates all CSV writing, chunk rotation, and file buffering logic
to keep the experiment runner simple.
"""
import csv
import io
import time
from PyQt5 import QtCore
import numpy as np


CSV_CHUNK_TARGET_BYTES = 128 * 1024 * 1024
CSV_CHUNK_MIN_SEC = 1.0


class ChunkWriter:
    """Handles writing multi-channel data to chunked CSV files with automatic rotation."""

    def __init__(self, sample_rate, num_channels=1, channel_names=None, chunk_max_sec=300,
                 buffer_bytes=1024 * 1024, flush_interval_sec=1.0,
                 chunk_target_bytes=CSV_CHUNK_TARGET_BYTES):
        self.sample_rate = float(sample_rate)
        self.num_channels = int(num_channels)
        self.channel_names = list(channel_names) if channel_names else None
        self.chunk_max_sec = float(chunk_max_sec)
        self.buffer_bytes = int(buffer_bytes)
        self.flush_interval_sec = float(flush_interval_sec)
        self.chunk_target_bytes = int(chunk_target_bytes)

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
        self._chunk_start_t = 0.0
        self.raw_chunks_dir = None
        self.filename_prefix = None
        self._io_mutex = QtCore.QMutex()

    def _format_text(self, seg, times, marker_ids, marker_names, has_marker):
        # savetxt formats in C and is ~1.3x faster than np.char.mod here.
        n = int(seg.shape[1])
        matrix = np.empty((n, 1 + seg.shape[0]), dtype=np.float64)
        matrix[:, 0] = times
        matrix[:, 1:] = seg.T
        fmt = ['%.6f'] + ['%.4f'] * seg.shape[0]
        buf = io.StringIO()
        np.savetxt(buf, matrix, fmt=fmt, delimiter=',')
        text = buf.getvalue()
        if has_marker:
            return "\n".join(
                f"{line},{int(marker_ids[i])},{marker_names[i]}"
                for i, line in enumerate(text.splitlines())
            ) + "\n"
        # C-level replace avoids splitlines+join on every ordinary block.
        return text.replace("\n", ",0,\n")

    def _format_rows(self, seg, times, marker_ids, marker_names, has_marker):
        return self._format_text(
            seg, times, marker_ids, marker_names, has_marker
        ).splitlines()

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
                seg = arr[:, offset:offset + take]
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
                self.csv_file_handle.write(self._format_text(
                    seg, times, marker_ids, marker_names, has_marker
                ))

                self._chunk_samples_written += take
                offset += take

                if self._chunk_samples_written >= self.chunk_max_samples:
                    self._rotate_chunk_locked()

            self.csv_sample_counter += n_samples
            self._chunks_since_flush += 1
            now = time.perf_counter()
            if has_marker or force_flush or (now - self._last_csv_flush_t) >= self.flush_interval_sec:
                self.csv_file_handle.flush()
                self._chunks_since_flush = 0
                self._last_csv_flush_t = now

            if (self._chunk_samples_written > 0
                    and (now - self._chunk_start_t) >= CSV_CHUNK_MIN_SEC):
                written = self.csv_file_handle.tell()
                if written >= self.chunk_target_bytes:
                    self._rotate_chunk_locked()

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
        self._chunk_start_t = time.perf_counter()
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
