"""rhx_realtime_feed.device - Intan RHX device interface."""

from ._rhx_config import RHXConfig, GetSampleRateFailure
from ._rhx_device import IntanRHXDevice

__all__ = ["RHXConfig", "GetSampleRateFailure", "IntanRHXDevice"]