from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Protocol, Optional, List, Any, Union
import numpy as np


@dataclass
class ParamDef:
    name: str
    label: str
    dtype: str
    default: Any = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    choices: Optional[List[str]] = None


@dataclass
class DeviceOperation:
    name: str
    label: str
    default_duration: float = 1.0
    instantaneous: bool = False
    params: List[ParamDef] = field(default_factory=list)
    color: str = "#0078D4"
    # Runs the step. Signature: (runner, step_idx, device, device_name, params, duration)
    handler: Optional[Any] = None
    # Runs while the device is still stopped, before its group starts.
    prepare: Optional[Any] = None

    @property
    def operation_id(self):
        return self.name


@dataclass
class ChannelInfo:
    index: int
    name: str
    direction: str
    unit: str = ""
    range_min: float = 0.0
    range_max: float = 0.0
    enabled: bool = True


class Device(ABC):
    config_version = 1
    operation_aliases = {}
    # On-disk storage this device writes and replay readers it can drive.
    storage_format = "csv"
    storage_globs = ("chunk_*.csv",)
    replay_source_formats = frozenset()

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def device_type(self) -> str: ...

    @property
    @abstractmethod
    def connected(self) -> bool: ...

    @property
    @abstractmethod
    def sample_rate(self) -> Optional[float]: ...

    @property
    @abstractmethod
    def channels(self) -> List[ChannelInfo]: ...

    @property
    def input_channels(self) -> List[ChannelInfo]:
        return [c for c in self.channels if c.direction in ("input", "bidirectional")]

    @property
    def output_channels(self) -> List[ChannelInfo]:
        return [c for c in self.channels if c.direction in ("output", "bidirectional")]

    @abstractmethod
    def connect(self) -> bool: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def start_acquisition(self) -> None: ...

    @abstractmethod
    def stop_acquisition(self) -> None: ...

    @abstractmethod
    def read_data(self) -> Optional[np.ndarray]: ...

    @abstractmethod
    def write_output(self, channel_index: int, value: Union[float, bool]) -> None: ...

    @abstractmethod
    def trigger_action(self, channel_index: int) -> None: ...

    def configure(self, **kwargs) -> None:
        pass

    @classmethod
    def get_operations(cls) -> List[DeviceOperation]:
        return []

    @classmethod
    def get_config_params(cls) -> List[ParamDef]:
        return []

    @classmethod
    def migrate_config(cls, config: dict, from_version: int) -> dict:
        return dict(config or {})

    @classmethod
    def canonical_operation_id(cls, operation_id: str) -> str:
        return cls.operation_aliases.get(operation_id, operation_id)

    @classmethod
    def get_tab_class(cls):
        from leech.gui.tabs.base import DataTab
        return DataTab

    def create_output_sink(self, sample_rate, channel_names, raw_dir, file_prefix,
                           flush_interval_sec=1.0):
        """Open the storage sink for a stream; ``None`` means the device records
        natively via ``start_recording_session``."""
        from leech.workers.chunk_writer import ChunkWriter
        from leech.telemetry_logger import append_telemetry_line

        num_channels = len(channel_names) or 1
        text_rate = float(num_channels) * float(sample_rate) * 9.0
        if text_rate > 10e6:
            append_telemetry_line(
                f"storage | warn | CSV at {num_channels} ch x {sample_rate:g} Hz "
                f"is ~{text_rate / 1e6:.1f} MB/s; use native or HDF5 storage"
            )
        sink = ChunkWriter(
            sample_rate=sample_rate,
            num_channels=num_channels,
            channel_names=channel_names,
            chunk_max_sec=300.0,
            buffer_bytes=1024 * 1024,
            flush_interval_sec=flush_interval_sec,
        )
        return self._open_sink(sink, raw_dir, file_prefix)

    @staticmethod
    def _open_sink(sink, raw_dir, file_prefix):
        sink.raw_chunks_dir = raw_dir
        sink.filename_prefix = file_prefix
        sink._open_new_chunk_locked()
        return sink

    @classmethod
    def replay_source(cls, chunks_dir, fmt, **kwargs):
        """Return a replay reader for ``fmt``; raise if the device has none."""
        raise ValueError(f"{cls.__name__} has no replay source for {fmt!r}")


class OutputSink(Protocol):
    def append_data(self, arr: np.ndarray, marker_info: dict = None) -> None: ...
    def close(self) -> None: ...


def migrate_storage_key(config: dict) -> dict:
    """Rename the legacy ``storage_mode`` device config key."""
    result = dict(config or {})
    if "storage_mode" in result and "storage_format" not in result:
        result["storage_format"] = result.pop("storage_mode")
    return result
