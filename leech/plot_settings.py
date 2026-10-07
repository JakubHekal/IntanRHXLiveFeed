from PyQt5.QtCore import QSettings

from leech.device.intan_rhx.processing import (
    PSD_BUFFER_SEC as DEFAULT_PSDS,
    WAVEFORM_BUFFER_SEC as DEFAULT_WAVEFORM,
    SPIKE_BIN_SEC as DEFAULT_SPIKE_BIN,
)

_SETTINGS = QSettings("LEECH", "LEECH")
_RECENT_EXPERIMENT_KEY = "recent_experiment_path"


def load_plot_setting(key: str, default: int) -> int:
    return _SETTINGS.value(key, default, type=int)


def save_plot_setting(key: str, value: int):
    _SETTINGS.setValue(key, value)


def save_recent_experiment(path: str):
    _SETTINGS.setValue(_RECENT_EXPERIMENT_KEY, path)


def load_recent_experiment() -> str:
    return _SETTINGS.value(_RECENT_EXPERIMENT_KEY, "", type=str)


def load_list(key: str, default: list) -> list:
    raw = _SETTINGS.value(key, "")
    if not raw:
        return list(default)
    try:
        return [int(x) for x in str(raw).split(",")]
    except (ValueError, TypeError):
        return list(default)


def save_list(key: str, values: list):
    _SETTINGS.setValue(key, ",".join(str(v) for v in values))


def load_geometry() -> bytes:
    return _SETTINGS.value("layout/geometry")


def save_geometry(data: bytes):
    _SETTINGS.setValue("layout/geometry", data)