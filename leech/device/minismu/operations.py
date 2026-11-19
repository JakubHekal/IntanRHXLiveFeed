"""miniSMU operation handlers (force voltage/current, measure, configure)."""

import re
import time

import numpy as np

from leech.telemetry_logger import append_telemetry_line
from leech.workers.storage_worker import StorageWorker


def configure(runner, step_idx, device, device_name, params, duration):
    device.configure(**{k: v for k, v in params.items()
                        if k not in ("duration_s", "block_label", "_start")})
    time.sleep(0.1)


def measure(runner, step_idx, device, device_name, params, duration):
    value = device.read_data()
    print(f"[Runner] Measure on {device_name}: {value}")
    time.sleep(0.1)


def force_voltage(runner, step_idx, device, device_name, params, duration):
    device.configure(mode="FVMI")
    ch = params.get("channel", 1) - 1
    val = params.get("voltage", params.get("amplitude", 5.0))
    try:
        device.write_output(max(0, ch), val)
        time.sleep(duration)
    finally:
        try:
            device.write_output(max(0, ch), 0.0)
        except Exception:
            pass


def force_current(runner, step_idx, device, device_name, params, duration):
    ch = params.get("channel", 1) - 1
    current_nA = float(params.get("current", params.get("current_nA", -100000.0)))
    label = params.get("block_label", "ForceCurrent")

    device.configure(mode="FIMV")

    sr = getattr(device, "sample_rate", 1000.0) or 1000.0
    ch_labels = [c.name for c in getattr(device, "channels", [])]

    dev_raw_dir = runner._run_path / "raw" / runner._raw_device_key(device, device_name)
    dev_raw_dir.mkdir(parents=True, exist_ok=True)

    file_label = re.sub(r"[^A-Za-z0-9._-]+", "_", str(label)).strip("._") or "ForceCurrent"
    sink = device.create_output_sink(
        sample_rate=sr,
        channel_names=ch_labels,
        raw_dir=dev_raw_dir,
        file_prefix=f"{step_idx + 1:06d}_{file_label}",
    )

    storage = StorageWorker(sink)
    device.start_acquisition()
    append_telemetry_line(
        f"acq_start | {step_idx} | {device_name} | force_current | "
        f"current_nA={current_nA} ch={ch} dur={duration} label={label}"
    )
    first_chunk = True
    device.write_output(ch, current_nA / 1e9)

    deadline = time.perf_counter() + duration
    while time.perf_counter() < deadline and not runner._abort:
        if runner._paused:
            pause_start = time.perf_counter()
            while runner._paused and not runner._abort:
                time.sleep(0.1)
            deadline += time.perf_counter() - pause_start
        try:
            data = device.read_data()
        except Exception:
            data = None
        if data is not None:
            arr = np.asarray(data)
            if arr.ndim == 2 and arr.shape[1] > 0:
                marker_info = (
                    {0: {"id": step_idx + 1, "name": label}} if first_chunk else None
                )
                if not storage.submit(arr, marker_info=marker_info):
                    runner._report_error(device_name, f"Storage writer failed: {storage.error}")
                    break
                runner.data_received.emit(device_name, arr)
                first_chunk = False
        remaining = deadline - time.perf_counter()
        time.sleep(min(0.01, max(0, remaining)) if remaining > 0 else 0.01)

    storage_error = storage.close()
    if storage_error is not None and not runner._abort:
        runner._report_error(device_name, f"Storage writer failed: {storage_error}")
    append_telemetry_line(
        f"acq_end | {step_idx} | {device_name} | force_current | "
        f"samples_in_chunk={sink._chunk_samples_written} backpressure={storage.backpressure_count}"
    )
    try:
        device.write_output(ch, 0.0)
    except Exception:
        pass
    try:
        device.stop_acquisition()
    except Exception:
        pass
