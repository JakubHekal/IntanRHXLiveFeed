from PyQt5.QtCore import QSettings

from rhx_realtime_feed.device.intan_rhx.processing import (
    PSD_BUFFER_SEC as DEFAULT_PSDS,
    WAVEFORM_BUFFER_SEC as DEFAULT_WAVEFORM,
    SPIKE_BIN_SEC as DEFAULT_SPIKE_BIN,
)

_SETTINGS = QSettings("RHX", "RealtimeFeed")
_RECENT_EXPERIMENT_KEY = "recent_experiment_path"


def load_plot_setting(key: str, default: int) -> int:
    return _SETTINGS.value(key, default, type=int)


def save_plot_setting(key: str, value: int):
    _SETTINGS.setValue(key, value)


def save_recent_experiment(path: str):
    _SETTINGS.setValue(_RECENT_EXPERIMENT_KEY, path)


def load_recent_experiment() -> str:
    return _SETTINGS.value(_RECENT_EXPERIMENT_KEY, "", type=str)