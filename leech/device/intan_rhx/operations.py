"""Intan-specific operation handlers (stimulation).

Kept out of the global runner so a new device can add actions without
touching experiment sequencing.
"""

import time

from leech.telemetry_logger import append_telemetry_line


def _stim_params(params):
    return {k: v for k, v in params.items() if k not in ("duration_s", "block_label", "_start")}


def prepare_stimulus(runner, step_idx, device, device_name, params, duration):
    """Upload the train while the board is stopped; uploads while running wedge RHX."""
    try:
        device.program_stimulation(_stim_params(params))
        append_telemetry_line(
            f"stim_programmed | {step_idx} | {device_name} | Stimulus | ok"
        )
    except Exception as exc:
        runner._report_error(device_name, str(exc))
        append_telemetry_line(
            f"step_error | {step_idx} | {device_name} | Stimulus | prepare: {exc}"
        )
        print(f"[Runner] Error preparing step {step_idx} (Stimulus): {exc}")


def run_stimulus(runner, step_idx, device, device_name, params, duration):
    if not getattr(device, "connected", False):
        runner._report_error(device_name, "Device not connected")
        return
    label = params.get("block_label", "Stimulus")
    append_telemetry_line(
        f"stim_start | {step_idx} | {device_name} | {_stim_params(params)} "
        f"dur={duration}s label={label}"
    )
    burst_on = float(params.get("burst_on_s", 0.0) or 0.0)
    burst_off = float(params.get("burst_off_s", 0.0) or 0.0)
    period_us = float(params.get("pulse_period_us", 200.0) or 200.0)
    train_dur = getattr(device, "_stim_train_duration_s", None)
    if train_dur is None:
        ppt = getattr(device, "_stim_pulses_per_train", 0)
        train_dur = ppt * period_us * 1e-6
    interval = float(params.get("train_interval_s", 0.0) or 0.0)
    if interval > 0:
        if interval < train_dur:
            append_telemetry_line(
                f"stim_warn | {step_idx} | {device_name} | Stimulus | "
                f"train_interval {interval:.3f}s shorter than train "
                f"{train_dur:.3f}s; clamped"
            )
            print(
                f"[Runner] Warning: train interval {interval}s < train duration "
                f"{train_dur:.3f}s, clamped"
            )
        interval = max(interval, train_dur)
    else:
        interval = train_dur
    burst_cycle = (burst_on + burst_off) if (burst_on > 0 and burst_off > 0) else 0.0
    # The Stream step sets runmode=run from a daemon thread; the first edge
    # written before the chip is clocked is silently lost. Wait for the
    # board to actually run before firing the first train.
    if getattr(device, "wait_for_run_mode", None) is not None:
        if not device.wait_for_run_mode(timeout=10.0):
            msg = "Stimulus aborted: device did not reach run mode before block start"
            runner._report_error(device_name, msg)
            append_telemetry_line(
                f"stim_error | {step_idx} | {device_name} | Stimulus | {msg}"
            )
            print(f"[Runner] {msg}")
            return
    t0 = time.perf_counter()
    deadline = t0 + duration
    next_train = t0
    trains = 0
    try:
        while time.perf_counter() < deadline and not runner._abort:
            if runner._paused:
                pause_start = time.perf_counter()
                while runner._paused and not runner._abort:
                    time.sleep(0.1)
                pause = time.perf_counter() - pause_start
                deadline += pause
                next_train += pause
                t0 += pause
            elapsed = time.perf_counter() - t0
            in_burst = burst_cycle <= 0 or (elapsed % burst_cycle) < burst_on
            if in_burst and time.perf_counter() >= next_train:
                device.trigger_train()
                trains += 1
                next_train += interval
            remaining = deadline - time.perf_counter()
            time.sleep(min(0.01, max(0, remaining)) if remaining > 0 else 0.01)
    finally:
        try:
            device.stop_stimulation()
        except Exception:
            pass
    ppt = getattr(device, "_stim_pulses_per_train", 0)
    pulses = trains * ppt
    append_telemetry_line(
        f"stim_end | {step_idx} | {device_name} | Stimulus | "
        f"{pulses} pulses ({trains} trains x {ppt})"
    )
    print(f"[Runner] Stimulus {label}: {pulses} pulses delivered ({trains} trains x {ppt})")
