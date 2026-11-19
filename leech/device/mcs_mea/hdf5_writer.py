"""HDF5 sink writing MCS RawData protocol v3 recordings.

Each rotated file is a standalone recording that McsPy and MCS tools can
open. Raw ADC counts are reconstructed from the microvolt chunks using the
hardware step, so the stored values reconstruct to the same microvolts.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

_ZERO_GUID = "00000000-0000-0000-0000-000000000000"


class Hdf5ChunkWriter:
    """Write ``(channels, samples)`` microvolt chunks to rotated HDF5 files."""

    def __init__(
        self,
        sample_rate: float,
        num_channels: int = 1,
        channel_names: Optional[List[str]] = None,
        chunk_max_sec: float = 300.0,
        flush_interval_sec: float = 1.0,
        adc_step_uv: float = 1.0,
        adc_bits: int = 24,
        mea_name: Optional[str] = None,
        mea_layout: Optional[str] = None,
    ) -> None:
        self.sample_rate = float(sample_rate)
        self.num_channels = int(num_channels)
        self.channel_names = list(channel_names or [f"CH{i + 1:03d}" for i in range(self.num_channels)])
        if len(self.channel_names) != self.num_channels:
            raise ValueError(
                f"HDF5 writer got {len(self.channel_names)} channel names for {self.num_channels} channels"
            )
        self.adc_step_uv = float(adc_step_uv)
        if not self.adc_step_uv > 0:
            raise ValueError("HDF5 writer adc_step_uv must be positive")
        self.adc_bits = int(adc_bits)
        self.mea_name = str(mea_name or f"MEA{self.num_channels}")
        self.mea_layout = str(mea_layout or f"MEA{self.num_channels}")
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
        self._timestamps: Any = None
        self._recording: Any = None
        self._markers_sample_index: Any = None
        self._markers_id: Any = None
        self._markers_name: Any = None
        self._last_flush = 0.0

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

        raw = np.rint(np.asarray(data, dtype=np.float64) / self.adc_step_uv).astype(np.int32)
        samples = int(data.shape[1])
        old = int(self._data.shape[1])
        self._data.resize((self.num_channels, old + samples))
        self._data[:, old:old + samples] = raw

        self._append_markers(marker_info)
        self._chunk_samples_written += samples
        self._total_samples += samples
        self._timestamps[0, 2] = self._chunk_samples_written - 1
        if force_flush or (time.monotonic() - self._last_flush) >= self.flush_interval_sec:
            self._flush_locked()

    def close(self) -> None:
        self._flush_locked()
        self._close_file()

    def _open_new_chunk_locked(self) -> None:
        import h5py

        if self.raw_chunks_dir is None:
            raise RuntimeError("HDF5 raw_chunks_dir is not set")
        self.raw_chunks_dir.mkdir(parents=True, exist_ok=True)
        self._current_chunk_path = (
            self.raw_chunks_dir / f"chunk_{self.filename_prefix}_{self._chunk_index:06d}.h5"
        )
        self._raw_chunk_paths.append(self._current_chunk_path)
        self._chunk_start_sample = self._total_samples
        start_us = int(round(self._chunk_start_sample * 1_000_000 / self.sample_rate))
        now = datetime.now(timezone.utc)
        date_ticks = int((now.timestamp() + 62135596800) * 10_000_000)

        self._file = h5py.File(self._current_chunk_path, "w")
        self._file.attrs.update({
            "McsHdf5ProtocolType": np.bytes_(b"RawData"),
            "McsHdf5ProtocolVersion": np.int32(3),
            "GeneratingApplicationName": np.bytes_(b"LEECH"),
            "GeneratingApplicationVersion": np.bytes_(self._app_version().encode("ascii")),
            "McsDataToolsVersion": np.bytes_(b"1.0.0"),
        })
        data_grp = self._file.create_group("Data")
        data_grp.attrs.update({
            "ProgramName": np.bytes_(b"LEECH"),
            "ProgramVersion": np.bytes_(self._app_version().encode("ascii")),
            "MeaName": np.bytes_(self.mea_name.encode("ascii")),
            "MeaLayout": np.bytes_(self.mea_layout.encode("ascii")),
            "MeaSN": np.bytes_(b""),
            "Date": np.bytes_(now.strftime("%A, %B %d, %Y").encode("ascii")),
            "DateInTicks": np.int64(date_ticks),
            "FileGUID": np.bytes_(str(uuid.uuid4()).encode("ascii")),
            "Comment": np.bytes_(b""),
        })
        self._recording = data_grp.create_group("Recording_0")
        self._recording.attrs.update({
            "RecordingID": np.int32(0),
            "RecordingType": np.bytes_(b""),
            "TimeStamp": np.int64(start_us),
            "Duration": np.int64(start_us),
            "Label": np.bytes_(b""),
            "Comment": np.bytes_(b""),
        })
        stream = self._recording.create_group("AnalogStream").create_group("Stream_0")
        stream.attrs.update({
            "StreamInfoVersion": np.int32(1),
            "Label": np.bytes_(b"Electrode Raw Data"),
            "SourceStreamGUID": np.bytes_(_ZERO_GUID.encode("ascii")),
            "StreamGUID": np.bytes_(str(uuid.uuid4()).encode("ascii")),
            "StreamType": np.bytes_(b"Analog"),
            "DataSubType": np.bytes_(b"Electrode"),
        })

        tick = int(round(1_000_000 / self.sample_rate))
        conversion = int(round(self.adc_step_uv * 1_000_000))
        if tick <= 0 or conversion <= 0:
            raise ValueError("Sample rate or ADC step is out of range for MCS metadata")
        self._write_info_channel(stream, tick, conversion)

        chunk_samples = max(1, 262_144 // (4 * self.num_channels))
        self._data = stream.create_dataset(
            "ChannelData",
            shape=(self.num_channels, 0),
            maxshape=(self.num_channels, None),
            dtype="int32",
            chunks=(self.num_channels, chunk_samples),
            compression="gzip",
            compression_opts=5,
            shuffle=True,
        )
        self._timestamps = stream.create_dataset(
            "ChannelDataTimeStamps", data=np.asarray([[start_us, 0, 0]], dtype=np.int64)
        )
        self._timestamps.attrs["Column-0"] = np.bytes_(b"FirstTimeStamp")
        self._timestamps.attrs["Column-1"] = np.bytes_(b"FirstIndex")
        self._timestamps.attrs["Column-2"] = np.bytes_(b"LastIndex")

        markers = self._file.create_group("markers")
        self._markers_sample_index = markers.create_dataset(
            "sample_index", shape=(0,), maxshape=(None,), dtype="int64"
        )
        self._markers_id = markers.create_dataset(
            "marker_id", shape=(0,), maxshape=(None,), dtype="int32"
        )
        self._markers_name = markers.create_dataset(
            "marker_name", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype(encoding="utf-8")
        )
        self._chunk_samples_written = 0
        self._last_flush = time.monotonic()

    def _write_info_channel(self, stream: Any, tick: int, conversion: int) -> None:
        import h5py

        string = h5py.string_dtype(encoding="ascii")
        dtype = np.dtype([
            ("ChannelID", "<i4"),
            ("RowIndex", "<i4"),
            ("GroupID", "<i4"),
            ("ElectrodeGroup", "<i4"),
            ("Label", string),
            ("RawDataType", string),
            ("Unit", string),
            ("Exponent", "<i4"),
            ("ADZero", "<i4"),
            ("Tick", "<i8"),
            ("ConversionFactor", "<i8"),
            ("ADCBits", "<i4"),
            ("HighPassFilterType", string),
            ("HighPassFilterCutOffFrequency", string),
            ("HighPassFilterOrder", "<i4"),
            ("LowPassFilterType", string),
            ("LowPassFilterCutOffFrequency", string),
            ("LowPassFilterOrder", "<i4"),
        ])
        records = np.zeros(self.num_channels, dtype=dtype)
        for index, label in enumerate(self.channel_names):
            records[index] = (
                index, index, 0, 0, str(label).encode("ascii"),
                b"Int", b"V", -12, 0, tick, conversion, self.adc_bits,
                b"", b"-1", -1, b"", b"-1", -1,
            )
        info = stream.create_dataset("InfoChannel", data=records)
        info.attrs["InfoVersion"] = np.int32(2)

    def _app_version(self) -> str:
        try:
            from leech import __version__
        except Exception:
            return "0.0.0"
        return str(__version__)

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
        if self._file is None:
            return
        if self._recording is not None:
            end_us = self._recording.attrs["TimeStamp"] + int(
                round(self._chunk_samples_written * 1_000_000 / self.sample_rate)
            )
            self._recording.attrs["Duration"] = np.int64(end_us)
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
        self._timestamps = None
        self._recording = None
        self._markers_sample_index = None
        self._markers_id = None
        self._markers_name = None
