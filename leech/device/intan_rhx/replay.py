"""Replay source adapter for native Intan .rhd/.rhs recordings."""

from pathlib import Path

import numpy as np

from .file_reader import _AMP_OFFSET, _AMP_SCALE_UV, native_files, read_intan_header


class IntanNativeSource:
    """Sequential reader for native Intan .rhd/.rhs recordings."""

    def __init__(
        self,
        chunks_dir: Path,
        sample_rate: float = 20000.0,
        label_order=None,
        device_type=None,
    ):
        del sample_rate, label_order, device_type  # header values are authoritative
        paths = native_files(chunks_dir)
        if not paths:
            raise FileNotFoundError(f"No Intan .rhd/.rhs files in {chunks_dir}")
        self._files = [read_intan_header(path) for path in paths]
        first = self._files[0]
        for other in self._files[1:]:
            if (
                abs(other.sample_rate - first.sample_rate) > 1e-6
                or other.num_channels != first.num_channels
            ):
                raise ValueError(
                    f"{first.path} and {other.path}: Intan files disagree on sample rate "
                    f"or amplifier channel count ({first.sample_rate} Hz/{first.num_channels} ch "
                    f"vs {other.sample_rate} Hz/{other.num_channels} ch)"
                )
        self._sample_rate = first.sample_rate
        self._labels = list(first.labels)
        self._num_channels = first.num_channels
        self._total = sum(item.total_samples for item in self._files)
        self._file_idx = 0
        self._block_idx = 0
        self._handle = None
        self._buffer = np.empty((self._num_channels, 0), dtype=np.float32)

    # ── metadata ────────────────────────────────────────────────────────
    @property
    def sample_rate(self) -> float:
        return self._sample_rate

    @property
    def num_channels(self) -> int:
        return self._num_channels

    @property
    def labels(self):
        return list(self._labels)

    @property
    def total_samples(self) -> int:
        return self._total

    @property
    def exhausted(self) -> bool:
        return self._file_idx >= len(self._files) and self._buffer.shape[1] == 0

    # ── reading ─────────────────────────────────────────────────────────
    def _close_handle(self):
        if self._handle is not None:
            try:
                self._handle.close()
            finally:
                self._handle = None

    def close(self):
        self._close_handle()
        self._file_idx = len(self._files)
        self._buffer = np.empty((self._num_channels, 0), dtype=np.float32)

    def _read_blocks(self, count: int):
        while self._file_idx < len(self._files):
            info = self._files[self._file_idx]
            remaining = info.num_blocks - self._block_idx
            if remaining <= 0:
                self._close_handle()
                self._file_idx += 1
                self._block_idx = 0
                continue
            count = min(count, remaining)
            if self._handle is None:
                self._handle = open(info.path, "rb")
                self._handle.seek(info.header_size)
            raw = self._handle.read(count * info.stride)
            if len(raw) != count * info.stride:
                self.close()
                raise ValueError(f"{info.path}: truncated Intan data block")
            blocks = np.frombuffer(raw, dtype=info.dtype, count=count)
            self._block_idx += count
            amp = blocks["amp"].transpose(1, 0, 2).reshape(info.num_channels, -1)
            chunk = amp.astype(np.float32)
            chunk -= np.float32(_AMP_OFFSET)
            chunk *= np.float32(_AMP_SCALE_UV)
            return chunk
        return None

    def read(self, n: int):
        if n <= 0 or self.exhausted:
            return None, []
        n = int(n)
        while self._buffer.shape[1] < n and not self.exhausted:
            needed = n - self._buffer.shape[1]
            samples_per_block = self._files[min(self._file_idx, len(self._files) - 1)].samples_per_block
            count = max(1, (needed + samples_per_block - 1) // samples_per_block)
            slab = max(1, int(round(self._sample_rate)) // samples_per_block)
            chunk = self._read_blocks(max(count, slab))
            if chunk is None:
                break
            if self._buffer.shape[1]:
                self._buffer = np.concatenate((self._buffer, chunk), axis=1)
            else:
                self._buffer = chunk
        if self._buffer.shape[1] == 0:
            return None, []
        take = min(n, self._buffer.shape[1])
        chunk = np.ascontiguousarray(self._buffer[:, :take])
        self._buffer = self._buffer[:, take:]
        return chunk, []
