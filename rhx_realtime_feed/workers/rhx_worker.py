from rhx_realtime_feed.workers.device_worker import DeviceWorker, RAW_CHUNK_SEC, CSV_FILE_BUFFER_BYTES, CSV_FLUSH_INTERVAL_SEC

CHANNELS_PORT = "B"
CHANNELS_TO_PLOT = [0]

PLOT_UPDATE_HZ = 20


class RHXWorker(DeviceWorker):
    pass
