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
from .tab import IntanDeviceTab

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

    def set_parameter(self, param, value):
        self.command_socket.sendall(f"set {param} {value}\n".encode())
        time.sleep(self.send_delay)

    def get_parameter(self, param):
        self.command_socket.sendall(f"get {param}\n".encode())
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
        self.command_socket.sendall(b"execute clearalldataoutputs\n")
        time.sleep(self.send_delay)

    def get_run_mode(self):
        response = self.get_parameter("runmode")
        return response.strip().split()[-1]

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
            self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.data_socket.connect((self.host, self.data_port))
            self.data_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
            try:
                self.data_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except Exception:
                pass
            self.data_socket.settimeout(0.005)
            self._connected = True
            self._sample_rate = self.get_sample_rate()
            self._sample_counter = 0
            self.effective_fs = float(self._sample_rate)
        except Exception as e:
            self._last_connect_error = str(e)
            self._connected = False
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
        _intentional_stop = True
        try:
            while self.streaming:
                rolling_buffer, peer_closed = self.receive_data(rolling_buffer, self.read_size)
                if peer_closed:
                    _intentional_stop = False
                    break
                emg_data, timestamps, consumed, self._synced = self.parse_emg_stream_fast(rolling_buffer)
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
        if stop_after_disconnect:
            if self.get_run_mode() == 'run':
                self.set_run_mode("stop")
        self.command_socket.close()
        self.data_socket.close()

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
            if n_new <= 0 or n_new > buf_len // 2:
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
