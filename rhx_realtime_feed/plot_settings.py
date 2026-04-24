from PyQt5.QtCore import QSettings

from rhx_realtime_feed.workers.processing_worker import (
    PSD_BUFFER_SEC as DEFAULT_PSDS,
    WAVEFORM_BUFFER_SEC as DEFAULT_WAVEFORM,
    SPIKE_BIN_SEC as DEFAULT_SPIKE_BIN,
)

_SETTINGS = QSettings("RHX", "RealtimeFeed")


def plot_settings() -> QSettings:
    return _SETTINGS


def load_plot_setting(key: str, default: int) -> int:
    return _SETTINGS.value(key, default, type=int)


def save_plot_setting(key: str, value: int):
    _SETTINGS.setValue(key, value)