"""
TCP interface for the Intan RHX system.

Provides IntanRHXDevice for connecting to and streaming data from
Intan RHD/RHS recording controllers over TCP/IP.
"""

import time
import socket
import numpy as np
import threading
from collections.abc import Iterable
from typing import Optional, List, Union

from ..base import Device, ChannelInfo
from .stim import build_stim_params, _fmt
from .tab import IntanDeviceTab
from leech.telemetry_logger import append_telemetry_line

FRAMES_PER_BLOCK = 128
MAGIC_NUMBER = 0x2ef07a08


class GetSampleRateFailure(Exception):
    pass


class IntanRHXDevice(Device):
    name = "Intan RHX"
    device_type = "rhx"
    _enabled_ports = []

    def __init__(self,
                 host="127.0.0.1",
                 command_port=5000,
                 data_port=5001,
                 num_channels=128,
                 buffer_duration_sec=5,
                 auto_start=False,
                 verbose=False):
        self.host = host
        self.command_port = command_port
        self.data_port = data_port
        self.num_channels = num_channels
        self._sample_rate = None
        self.verbose = verbose
        self._connected = False
        self.command_socket = None
        self.data_socket = None
        self.send_delay = 0.05
        self._cmd_lock = threading.Lock()

        self.buffer_duration_sec = buffer_duration_sec
        self.circular_buffer = None
        self.circular_idx = 0
        self._last_read_cursor = 0
        self.buffer_lock = threading.Lock()
        self.streaming_thread = None
        self.streaming = False

        self._enabled_channel_indices = list(range(num_channels))
        self.bytes_per_frame = 4 + 2 * self.num_channels
        self.bytes_per_block = 4 + FRAMES_PER_BLOCK * self.bytes_per_frame
        self.blocks_per_write = 1
        self.read_size = self.bytes_per_block * 1
        self._synced = False

        if auto_start:
            self.connect()
            if self._connected:
                self.start_streaming()

    # ── Device ABC property conformance ──

    @property
    def connected(self) -> bool:
        return self._connected

    @connected.setter
    def connected(self, value):
        self._connected = bool(value)

    @property
    def sample_rate(self) -> Optional[float]:
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        self._sample_rate = float(value) if value is not None else None

    def _send(self, cmd, delay=None):
        with self._cmd_lock:
            self.command_socket.sendall(cmd.rstrip("\n").encode() + b";\n")
            time.sleep(delay if delay is not None else self.send_delay)
            self._drain_command_reply()

    def _drain_command_reply(self):
        """Consume any async error/reply the server queued for a 'set' command,
        so the next 'get' doesn't read a stale line (e.g. 'Board must be running
        in order to stop')."""
        try:
            self.command_socket.settimeout(0.01)
            while True:
                try:
                    if not self.command_socket.recv(4096):
                        break
                except socket.timeout:
                    break
        except OSError:
            pass
        finally:
            self.command_socket.settimeout(2.0)

    def set_parameter(self, param, value):
        self._send(f"set {param} {value}\n")

    def get_parameter(self, param):
        with self._cmd_lock:
            self.command_socket.sendall(f"get {param};\n".encode())
            time.sleep(self.send_delay)
            return self.command_socket.recv(1024).decode()

    def enable_wide_channel(self, channels, port='a', status=True):
        if isinstance(channels, int):
            channels = [channels]
        elif not isinstance(channels, Iterable):
            raise TypeError("Channels must be an int, range, or iterable list.")
        for ch in channels:
            name = f"{port}-{ch:03d}"
            self.set_parameter(f"{name}.tcpdataoutputenabled", 'true' if status else 'false')

    def clear_all_data_outputs(self):
        self._send("execute clearalldataoutputs\n")

    def execute_command(self, cmd, delay=0.01):
        self._send(cmd, delay=delay)

    def _channel_names(self, channel_str):
        if str(channel_str).strip():
            indices = self._parse_channel_range(str(channel_str))
        else:
            indices = self._enabled_channel_indices
        return [f"{self._PORTS[idx // 32]}-{idx % 32:03d}" for idx in sorted(set(indices)) if idx // 32 < 4]

    _STIM_KEYS = ("shape", "polarity", "amplitude_uA", "second_amplitude_uA",
                  "phase_duration_us", "second_phase_duration_us", "interphase_delay_us",
                  "pulse_period_us", "refractory_period_us",
                  "pre_stim_amp_settle_us", "post_stim_amp_settle_us",
                  "pulses_per_train")

    def program_stimulation(self, params=None):
        """Program stim registers while the board is STOPPED.

        Never call while the board is running: the server's upload path
        (setStimSequenceParameters) does run() then spins on
        `while (isRunning()) qApp->processEvents()`, which never exits while
        streaming and freezes the server's FIFO draining (the 45 s data stall).
        """
        if self.get_run_mode() != 'stop':
            raise RuntimeError(
                "Cannot program stimulation while the Intan controller is running "
                "(uploading while running wedges the server run loop). Program "
                "stimulation in a prepare step before streaming starts."
            )
        kwargs = {k: params[k] for k in self._STIM_KEYS if params and k in params}
        stim_params, warnings = build_stim_params(**kwargs)
        p = dict(stim_params)
        channel_str = str((params or {}).get("channels", "") or "").strip()
        if not channel_str:
            raise ValueError(
                "No stimulation channels specified — set the 'Channels' field "
                "explicitly; stimulating all enabled channels is not allowed"
            )
        for w in warnings:
            append_telemetry_line(f"intanrhx | stim_warning | {w}")
            print(f"[IntanRHX] Warning: {w}")
        names = self._channel_names(channel_str)
        if not names:
            raise ValueError(f"No valid stimulation channels in '{channel_str}'")
        self._stim_trigger = "f1"
        for name in names:
            for param, value in stim_params:
                self.execute_command(f"set {name}.{param} {_fmt(value)}\n")
            self.execute_command(f"execute uploadstimparameters {name}\n")
        time.sleep(0.2)
        # Re-trigger cadence must cover the whole train plus the post-train
        # refractory tail, or the next edge lands while the chip is still busy
        # and is ignored (~5 s gaps between delivered trains).
        self._stim_pulses_per_train = int(p['NumberOfStimPulses'])
        self._stim_train_duration_s = (
            p['PulseTrainPeriodMicroseconds'] * self._stim_pulses_per_train
            + p['RefractoryPeriodMicroseconds']
        ) * 1e-6
        print(f"[IntanRHX] Stimulation programmed: ch={names} "
              f"amp={_fmt(p['FirstPhaseAmplitudeMicroAmps'])}uA "
              f"period={_fmt(p['PulseTrainPeriodMicroseconds'])}us "
              f"pulses/train={self._stim_pulses_per_train} "
              f"train={self._stim_train_duration_s:.2f}s")
        return p

    def trigger_train(self):
        """Fire one edge-triggered pulse train (up to 256 pulses) on the
        programmed channel. Hold the wire high ~50 ms so the chip sees a clean
        rising edge; the server's back-to-back `manualstimtriggerpulse` can
        coalesce into a sub-ms blip that never registers."""
        trigger = getattr(self, "_stim_trigger", "f1")
        self.execute_command(f"execute manualstimtriggeron {trigger}\n", delay=0.05)
        self.execute_command(f"execute manualstimtriggeroff {trigger}\n")

    def stop_stimulation(self):
        # Edge trains are self-terminating; nothing to hold off.
        print("[IntanRHX] Stimulation stopped")

    def get_run_mode(self):
        response = self.get_parameter("runmode")
        # real server replies "Return: RunMode Stop"/"Run" (capitalized)
        return response.strip().split()[-1].lower()

    def wait_for_run_mode(self, timeout=10.0):
        """Block until the board is actually running, or return False.

        The Stimulus step races the Stream step's runmode=run (set from a
        daemon thread), so the first trigger edge can be written while the chip
        is still unclocked and silently lost. Gate the first train on this."""
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            try:
                if self.get_run_mode() == 'run':
                    return True
            except Exception:
                pass
            time.sleep(0.05)
        return False

    def set_run_mode(self, mode):
        assert mode in ["run", "stop"], "Mode must be 'run' or 'stop'"
        self.set_parameter("runmode", mode)

    def get_sample_rate(self):
        resp = self.get_parameter("sampleratehertz")
        expected_return_string = "Return: SampleRateHertz "
        if resp.find(expected_return_string) == -1:
            raise GetSampleRateFailure('Unable to get sample rate from server.')
        if expected_return_string in resp:
            sample_rate = float(resp[len(expected_return_string):])
        else:
            raise ValueError(f"Unable to get sample rate from server: {resp}")
        return sample_rate

    def set_blocks_per_write(self, num_blocks):
        self.set_parameter("TCPNumberDataBlocksPerWrite", num_blocks)

    def connect(self):
        try:
            self.command_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.command_socket.connect((self.host, self.command_port))
            self.command_socket.settimeout(2.0)
            self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.data_socket.connect((self.host, self.data_port))
            self.data_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
            try:
                self.data_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except Exception:
                pass
            self.data_socket.settimeout(0.005)
            self._connected = True
            # ponytail: force a known-stopped controller before streaming; a board
            # left running by a previous failed close() desyncs blocksPerWrite,
            # runmode, and the TCP output geometry on the next session.
            try:
                self.set_run_mode("stop")
            except Exception:
                pass
            if self.get_run_mode() != 'stop':
                raise ConnectionError(
                    "Intan controller still running after 'set runmode stop' — "
                    "stop it in the Intan software (or close other sessions) and reconnect."
                )
            self._sample_rate = self.get_sample_rate()
            self._sample_counter = 0
            self.effective_fs = float(self._sample_rate)
        except Exception as e:
            self._last_connect_error = str(e)
            self._connected = False
            for sock in (self.command_socket, self.data_socket):
                try:
                    if sock is not None:
                        sock.close()
                except OSError:
                    pass
            self.command_socket = None
            self.data_socket = None
        return self._connected

    def receive_data(self, buffer: bytearray, read_size: int, max_reads: int = 16):
        peer_closed = False
        for _ in range(max_reads):
            try:
                chunk = self.data_socket.recv(read_size)
                if not chunk:
                    peer_closed = True
                    break
                buffer.extend(chunk)
                if len(chunk) < read_size:
                    break
            except socket.timeout:
                break
            except (ConnectionResetError, ConnectionAbortedError, OSError):
                peer_closed = True
                break
        return buffer, peer_closed

    @staticmethod
    def _parse_channel_range(s: str) -> list[int]:
        indices = []
        for part in s.split(','):
            part = part.strip()
            if not part:
                continue
            if '-' in part:
                a, b = part.split('-', 1)
                indices.extend(range(int(a), int(b) + 1))
            else:
                indices.append(int(part))
        return indices

    _PORTS = ['a', 'b', 'c', 'd']

    def _set_enabled_channels(self, indices: list[int]):
        indices = sorted(set(indices))
        self.clear_all_data_outputs()
        for idx in indices:
            port_idx = idx // 32
            ch = idx % 32
            if port_idx < 4:
                self.enable_wide_channel(ch, port=self._PORTS[port_idx])
        self._enabled_channel_indices = indices
        self.num_channels = len(indices)
        self._update_read_size()
        self.init_circular_buffer()

    def init_circular_buffer(self):
        buffer_length = int(self.sample_rate * self.buffer_duration_sec)
        self.circular_buffer = np.zeros((self.num_channels, buffer_length), dtype=np.float32)
        self.circular_idx = 0
        self._last_read_cursor = 0

    def parse_emg_stream_fast(self, raw_bytes: bytearray, synced=True):
        C = self.num_channels
        frames = FRAMES_PER_BLOCK
        bytes_per_frame = 4 + 2 * C
        bytes_per_block = 4 + frames * bytes_per_frame
        mv = memoryview(raw_bytes)
        start = 0
        if not synced:
            magic = MAGIC_NUMBER.to_bytes(4, "little")
            pos = raw_bytes.find(magic)
            if pos == -1:
                return None, None, 0, False
            start = pos
        available = (len(raw_bytes) - start)
        nblocks = available // bytes_per_block
        if nblocks <= 0:
            return None, None, start, True
        frame_dtype = np.dtype([('ts', '<i4'), ('v', ('<u2', C))])
        block_dtype = np.dtype([('magic', '<u4'), ('data', (frame_dtype, frames))])
        blocks = np.frombuffer(mv[start:start + nblocks * bytes_per_block], dtype=block_dtype)
        good = (blocks['magic'] == MAGIC_NUMBER)
        blocks = blocks[good]
        if blocks.size == 0:
            return None, None, start + nblocks * bytes_per_block, True
        frames_arr = blocks['data'].reshape(-1)
        ts = frames_arr['ts'].astype(np.int64)
        v = frames_arr['v'].astype(np.int32, copy=False)
        v -= 32768
        v *= 195
        emg = (v.astype(np.float32, copy=False) / 1000.0).T
        consumed = start + nblocks * bytes_per_block
        return emg, ts, consumed, True

    def parse_emg_stream(self, raw_bytes, return_all_timestamps=True):
        from struct import unpack
        C = self.num_channels
        idx = 0
        timestamps = []
        channel_data = [[] for _ in range(self.num_channels)]
        bytes_per_sample = 4 + 2 * C
        bytes_per_block = 4 + FRAMES_PER_BLOCK * bytes_per_sample
        while idx + bytes_per_block <= len(raw_bytes):
            if unpack('<I', raw_bytes[idx:idx + 4])[0] != MAGIC_NUMBER:
                idx += 1
                continue
            idx += 4
            for _ in range(FRAMES_PER_BLOCK):
                ts = unpack('<i', raw_bytes[idx:idx + 4])[0]
                if return_all_timestamps:
                    timestamps.append(ts)
                last_ts = ts
                idx += 4
                for ch in range(self.num_channels):
                    val = unpack('<H', raw_bytes[idx:idx + 2])[0]
                    voltage = 0.195 * (val - 32768)
                    channel_data[ch].append(voltage)
                    idx += 2
        emg_array = np.array(channel_data, dtype=np.float32)
        bytes_consumed = idx
        if return_all_timestamps:
            return emg_array, np.array(timestamps, dtype=np.int64), bytes_consumed
        else:
            return emg_array, last_ts if 'last_ts' in locals() else None, bytes_consumed

    def _streaming_worker(self):
        rolling_buffer = bytearray()
        self.set_run_mode("run")
        # ponytail: bounded drain of stale TCP data (<=200 ms) so the parser
        # doesn't misalign on leftover bytes from a previous stream geometry.
        # Never unbounded — it would eat live data forever when the controller
        # is already running (runmode set is a no-op then).
        drain_deadline = time.monotonic() + 0.2
        try:
            self.data_socket.settimeout(0.01)
            while time.monotonic() < drain_deadline:
                if not self.data_socket.recv(self.read_size):
                    break
        except socket.timeout:
            pass
        except (ConnectionResetError, ConnectionAbortedError, OSError):
            pass
        finally:
            self.data_socket.settimeout(0.005)
        self._synced = False
        _intentional_stop = True
        try:
            while self.streaming:
                rolling_buffer, peer_closed = self.receive_data(rolling_buffer, self.read_size)
                if peer_closed:
                    _intentional_stop = False
                    break
                emg_data, timestamps, consumed, self._synced = self.parse_emg_stream_fast(rolling_buffer, synced=self._synced)
                rolling_buffer = rolling_buffer[consumed:]
                if emg_data is not None:
                    n = emg_data.shape[1]
                    with self.buffer_lock:
                        idx = self.circular_idx
                        buf_len = self.circular_buffer.shape[1]
                        if n >= buf_len:
                            self.circular_buffer[:,:] = emg_data[:, -buf_len:]
                            self.circular_idx = 0
                        else:
                            end_idx = idx + n
                            if end_idx < buf_len:
                                self.circular_buffer[:, idx:end_idx] = emg_data
                            else:
                                part1 = buf_len - idx
                                self.circular_buffer[:, idx:] = emg_data[:, :part1]
                                self.circular_buffer[:, :n - part1] = emg_data[:, part1:]
                            self.circular_idx = (idx + n) % buf_len
        except:
            _intentional_stop = False
        finally:
            try:
                self.set_run_mode("stop")
            except Exception:
                pass
            self.streaming = False
            if not _intentional_stop:
                self._connected = False

    def _update_read_size(self):
        self.bytes_per_frame = 4 + 2 * self.num_channels
        self.bytes_per_block = 4 + FRAMES_PER_BLOCK * self.bytes_per_frame
        self.read_size = self.bytes_per_block * max(1, int(getattr(self, "blocks_per_write", 1)))

    def configure(self, **kwargs):
        for key, value in kwargs.items():
            if "channels" in key:
                if str(value).strip():
                    indices = self._parse_channel_range(str(value))
                    self._set_enabled_channels(indices)
            elif "blocks_per_write" in key:
                self.set_blocks_per_write(value)
                self.blocks_per_write = max(1, int(value))
                self._update_read_size()
            elif "enable_wide_channel" in key:
                port = kwargs.get("port", "a")
                self._enabled_ports = [port]
                self.enable_wide_channel(value, port=port)
                self._update_read_size()
            elif "port" in key:
                pass

    def start_streaming(self):
        if self.streaming:
            return
        if self._sample_rate is None:
            self._sample_rate = self.get_sample_rate()
        self.effective_fs = float(self._sample_rate)
        self.init_circular_buffer()
        try:
            self.set_blocks_per_write(1)
            self.blocks_per_write = 1
            self._update_read_size()
            resp = self.get_parameter("TCPNumberDataBlocksPerWrite")
            if resp.strip().split()[-1] != "1":
                append_telemetry_line(
                    f"intanrhx | warn | TCPNumberDataBlocksPerWrite="
                    f"{resp.strip().split()[-1]} (expected 1) — server desynced"
                )
                print("[IntanRHX] Warning: TCPNumberDataBlocksPerWrite != 1 on server")
        except Exception:
            pass
        self._connected = True
        self.streaming = True
        self.streaming_thread = threading.Thread(target=self._streaming_worker, daemon=True)
        self.streaming_thread.start()

    def stop_streaming(self):
        self.streaming = False
        if self.streaming_thread is not None:
            self.streaming_thread.join()
            self.streaming_thread = None

    def get_latest_window(self, duration_ms=200):
        num_samples = int(self.sample_rate * duration_ms / 1000)
        buf_len = self.circular_buffer.shape[1]
        with self.buffer_lock:
            idx = self.circular_idx
            if num_samples > buf_len:
                raise ValueError("Requested window exceeds buffer size")
            start_idx = (idx - num_samples) % buf_len
            if start_idx < idx:
                window = self.circular_buffer[:, start_idx:idx]
            else:
                window = np.hstack([self.circular_buffer[:, start_idx:], self.circular_buffer[:, :idx]])
        return window

    def get_latest_window_with_cursor(self, duration_ms=200):
        num_samples = int(self.sample_rate * duration_ms / 1000)
        buf_len = self.circular_buffer.shape[1]
        with self.buffer_lock:
            idx = int(self.circular_idx)
            if num_samples > buf_len:
                raise ValueError("Requested window exceeds buffer size")
            start_idx = (idx - num_samples) % buf_len
            if start_idx < idx:
                window = self.circular_buffer[:, start_idx:idx]
            else:
                window = np.hstack([self.circular_buffer[:, start_idx:], self.circular_buffer[:, :idx]])
        return window, idx

    def record(self, duration_sec=10, verbose=True):
        total_samples = int(self.sample_rate * duration_sec)
        collected_emg = np.zeros((self.num_channels, total_samples), dtype=np.float32)
        write_index = 0
        rolling_buffer = bytearray()
        sample_counter = 0
        last_print = time.time()
        self.set_run_mode("run")
        try:
            while write_index < total_samples:
                rolling_buffer, _ = self.receive_data(rolling_buffer, self.read_size)
                emg_data, timestamps, consumed, self._synced = self.parse_emg_stream_fast(
                    rolling_buffer, synced=self._synced
                )
                if consumed:
                    del rolling_buffer[:consumed]
                if emg_data is not None:
                    n = emg_data.shape[1]
                    store = min(n, total_samples - write_index)
                    collected_emg[:, write_index:write_index + store] = emg_data[:, :store]
                    write_index += store
                    sample_counter += store
                now = time.time()
                if now - last_print >= 1.0 and verbose:
                    rate = sample_counter / (now - last_print)
                    sample_counter = 0
                    last_print = now
        finally:
            self.set_run_mode("stop")
            return collected_emg

    def close(self, stop_after_disconnect=True):
        # ponytail: idempotent + exception-safe; a double close (runner finally
        # vs UI _close_devices) previously threw WinError 10038 and skipped the
        # 'set runmode stop', leaving the board running for the next session.
        if self.command_socket is None and self.data_socket is None:
            return
        if stop_after_disconnect and self.command_socket is not None:
            try:
                if self.get_run_mode() != 'stop':
                    self.set_run_mode("stop")
            except Exception:
                pass
        for sock in (self.command_socket, self.data_socket):
            try:
                if sock is not None:
                    sock.close()
            except OSError:
                pass
        self.command_socket = None
        self.data_socket = None
        self._connected = False
        self.streaming = False

    def record_to_file(self, path, duration_sec=10):
        emg = self.record(duration_sec)
        np.savez(path, emg=emg, sample_rate=self.sample_rate)

    @property
    def channels(self) -> List[ChannelInfo]:
        indices = getattr(self, '_enabled_channel_indices', list(range(self.num_channels)))
        return [
            ChannelInfo(idx, f"{self._PORTS[idx // 32].upper()}-{idx % 32:03d}", "input", "uV", -5000.0, 5000.0)
            for idx in indices if idx // 32 < 4
        ]

    def start_acquisition(self) -> None:
        self.start_streaming()

    def stop_acquisition(self) -> None:
        self.stop_streaming()

    def read_data(self) -> Optional[np.ndarray]:
        if self.circular_buffer is None:
            return None
        with self.buffer_lock:
            cur = int(self.circular_idx)
            prev = int(self._last_read_cursor)
            buf_len = self.circular_buffer.shape[1]
            n_new = (cur - prev) % buf_len
            if n_new <= 0:
                return None
            if n_new > buf_len // 2:
                # ponytail: advance cursor so next call sees n_new=0, not same
                # large value — breaks permanent stall after reconfigure
                self._last_read_cursor = cur
                return None
            if prev < cur:
                data = self.circular_buffer[:, prev:cur].copy()
            else:
                data = np.hstack([
                    self.circular_buffer[:, prev:],
                    self.circular_buffer[:, :cur],
                ]).copy()
            self._last_read_cursor = cur
            return data

    def write_output(self, channel_index: int, value: Union[float, bool]) -> None:
        pass

    def trigger_action(self, channel_index: int) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @classmethod
    def get_operations(cls):
        from ..base import DeviceOperation, ParamDef
        return [
            DeviceOperation("Configure", "Configure", instantaneous=True, default_duration=0, color="#2B88D8", params=[
                ParamDef("blocks_per_write", "Blocks per Write", "int", default=1, min_val=1, max_val=100),
                ParamDef("enable_wide_channel", "Enable Wide Channel", "bool", default=False),
            ]),
            DeviceOperation("Stream", "Stream", default_duration=10.0, color="#4BA3E3", params=[
                ParamDef("channels", "Channels (e.g. 0-31)", "channel_list", default=""),
            ]),
            DeviceOperation("Stimulus", "Stimulation", default_duration=1200.0, color="#E67E22", params=[
                ParamDef("channels", "Channels (e.g. 0-31)", "channel_list", default=""),
                ParamDef("shape", "Shape", "choice", default="Biphasic",
                         choices=["Biphasic", "BiphasicWithInterphaseDelay", "Triphasic"]),
                ParamDef("polarity", "Polarity", "choice", default="NegativeFirst",
                         choices=["NegativeFirst", "PositiveFirst"]),
                ParamDef("amplitude_uA", "First Phase Amplitude (uA)", "float", default=0.5, min_val=0.0, max_val=2550.0),
                ParamDef("second_amplitude_uA", "Second Phase Amplitude (uA)", "float", default=0.5, min_val=0.0, max_val=2550.0),
                ParamDef("phase_duration_us", "First Phase Duration (us)", "float", default=100.0, min_val=1.0, max_val=5000.0),
                ParamDef("second_phase_duration_us", "Second Phase Duration (us)", "float", default=100.0, min_val=1.0, max_val=5000.0),
                ParamDef("interphase_delay_us", "Interphase Delay (us)", "float", default=0.0, min_val=0.0, max_val=5000.0),
                ParamDef("pulse_period_us", "Pulse Period (us)", "float", default=200.0, min_val=1.0, max_val=1000000.0),
                ParamDef("refractory_period_us", "Refractory Period (us)", "float", default=0.0, min_val=0.0, max_val=1000000.0),
                ParamDef("pulses_per_train", "Pulses per Train (1-256)", "int", default=256, min_val=1, max_val=256),
                ParamDef("pre_stim_amp_settle_us", "Pre-Stim Amp Settle (us)", "float", default=0.0, min_val=0.0, max_val=500000.0),
                ParamDef("post_stim_amp_settle_us", "Post-Stim Amp Settle (us)", "float", default=0.0, min_val=0.0, max_val=500000.0),
                ParamDef("burst_on_s", "Burst On (s, 0 = continuous)", "float", default=0.0, min_val=0.0, max_val=100000.0),
                ParamDef("burst_off_s", "Burst Off (s, 0 = continuous)", "float", default=0.0, min_val=0.0, max_val=100000.0),
                ParamDef("train_interval_s", "Train Interval (s, 0 = fastest)", "float", default=0.0, min_val=0.0, max_val=100000.0),
            ])
        ]

    @classmethod
    def get_config_params(cls):
        from ..base import ParamDef
        return [
            ParamDef("host", "Host", "str", default="127.0.0.1"),
            ParamDef("command_port", "Command Port", "int", default=5000, min_val=1024, max_val=65535),
            ParamDef("data_port", "Data Port", "int", default=5001, min_val=1024, max_val=65535),
            ParamDef("num_channels", "Num Channels", "int", default=128, min_val=1, max_val=512),
            ParamDef("buffer_duration_sec", "Buffer Duration (s)", "float", default=5.0, min_val=1.0, max_val=60.0),
        ]

    @classmethod
    def get_tab_class(cls):
        return IntanDeviceTab
