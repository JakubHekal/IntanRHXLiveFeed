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


class OutputSink(Protocol):
    @property
    def sample_index(self) -> int: ...

    def start_session(self, name: str, project_path: str = None) -> dict: ...
    def append_data(self, arr: np.ndarray, marker_info: dict = None) -> None: ...
    def close(self) -> None: ...
    def get_paths(self) -> dict: ...
