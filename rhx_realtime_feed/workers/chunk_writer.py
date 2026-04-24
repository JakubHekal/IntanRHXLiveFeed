"""
ChunkWriter: Manages CSV chunk file I/O, rotation, and persistence.

Encapsulates all CSV writing, chunk rotation, and file buffering logic
to reduce RHXWorker complexity.
"""
import csv
import time
import re
from pathlib import Path
from PyQt5 import QtCore
import numpy as np


class ChunkWriter:
    """Handles writing EMG data to chunked CSV files with automatic rotation."""
    
    def __init__(self, sample_rate, channel=0, chunk_max_sec=300, buffer_bytes=1024*1024, flush_interval_sec=1.0):
        """
        Initialize the chunk writer.
        
        Args:
            sample_rate: Sampling rate in Hz.
            channel: Channel index being written.
            chunk_max_sec: Maximum duration (seconds) per CSV chunk file.
            buffer_bytes: File buffer size in bytes.
            flush_interval_sec: Auto-flush interval in seconds.
        """
        self.sample_rate = float(sample_rate)
        self.channel = int(channel)
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
        
        self._io_mutex = QtCore.QMutex()
    
    def start_session(self, project_name, project_path):
        """
        Create project directories and initialize CSV writer.
        
        Args:
            project_name: Name of the project (for directory naming).
            project_path: Base directory path for recordings.
        
        Returns:
            Tuple of (project_run_dir, raw_chunks_dir).
        """
        with QtCore.QMutexLocker(self._io_mutex):
            safe_project = re.sub(r"[^A-Za-z0-9._-]+", "_", str(project_name).strip()) or "project"
            timestamp = time.time()
            from datetime import datetime
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            base_dir = Path(project_path).expanduser() if project_path else (Path.cwd() / "recordings")
            base_dir.mkdir(parents=True, exist_ok=True)
            
            project_run_dir = base_dir / f"{safe_project}_{timestamp_str}"
            raw_chunks_dir = project_run_dir / "raw_chunks"
            project_run_dir.mkdir(parents=True, exist_ok=True)
            raw_chunks_dir.mkdir(parents=True, exist_ok=True)
            
            self.raw_chunks_dir = raw_chunks_dir
            self.project_run_dir = project_run_dir
            self._chunk_index = 0
            self._chunk_samples_written = 0
            self._raw_chunk_paths = []
            self._current_chunk_path = None
            self.csv_sample_counter = 0
            self._chunks_since_flush = 0
            self._last_csv_flush_t = time.perf_counter()
            
            self._open_new_chunk_locked()
            
            return project_run_dir, raw_chunks_dir
    
    def append_data(self, arr: np.ndarray, marker_info: dict = None, force_flush: bool = False):
        """
        Append a chunk of data (with optional markers) to the CSV.
        
        Args:
            arr: Data array of shape (channels, samples).
            marker_info: Dictionary mapping sample_index -> marker record. Defaults to no markers.
            force_flush: Force flush to disk even if not due yet.
        """
        if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 1:
            return
        
        with QtCore.QMutexLocker(self._io_mutex):
            if self.csv_file_handle is None:
                return
            
            values_all = arr[0, :].astype(np.float64, copy=False)
            n_samples = int(values_all.size)
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
                seg_end = int(seg_start + take)
                seg_vals = values_all[offset:offset + take]
                
                if not has_marker:
                    # Fast path: no markers, direct writelines
                    rows = [
                        f"{(seg_start + i) * inv_fs:.6f},{float(seg_vals[i]):.4f},0,\n"
                        for i in range(take)
                    ]
                    self.csv_file_handle.writelines(rows)
                    self._chunk_samples_written += take
                    offset += take
                    
                    if self._chunk_samples_written >= self.chunk_max_samples:
                        self._rotate_chunk_locked()
                    continue
                
                # Slow path: markers present, use writerows with marker lookups
                marker_ids = np.zeros(take, dtype=np.int64)
                marker_names = [""] * take
                for sample_idx, marker_rec in marker_info.items():
                    if sample_idx < seg_start or sample_idx >= seg_end:
                        continue
                    local = sample_idx - seg_start
                    marker_ids[local] = int(marker_rec.get("id", 0))
                    marker_names[local] = str(marker_rec.get("name", ""))
                
                rows = [
                    [
                        f"{(seg_start + i) * inv_fs:.6f}",
                        f"{float(seg_vals[i]):.4f}",
                        int(marker_ids[i]),
                        marker_names[i],
                    ]
                    for i in range(take)
                ]
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
        """Open a new CSV chunk file (must hold _io_mutex)."""
        if self.raw_chunks_dir is None:
            return
        self._chunk_index += 1
        path = self.raw_chunks_dir / f"chunk_{self._chunk_index:06d}.csv"
        self.csv_file_handle = open(
            path,
            "w",
            newline="",
            encoding="utf-8",
            buffering=self.buffer_bytes,
        )
        self.csv_writer = csv.writer(self.csv_file_handle)
        self.csv_writer.writerow(["time_s", f"ch_{self.channel}_uV", "marker_id", "marker_name"])
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
    
    def get_paths(self):
        """Return current project paths (thread-safe snapshot)."""
        with QtCore.QMutexLocker(self._io_mutex):
            return {
                "raw_chunks_dir": str(self.raw_chunks_dir) if self.raw_chunks_dir else "",
                "current_chunk": str(self._current_chunk_path) if self._current_chunk_path else "",
                "chunks": list(self._raw_chunk_paths),
            }
