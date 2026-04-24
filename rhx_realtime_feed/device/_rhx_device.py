"""
rhx_realtime_feed.device._rhx_device

TCP interface for the Intan RHX system.

This module provides the IntanRHXDevice class for connecting to and streaming
data from Intan RHD/RHS recording controllers over TCP/IP.
"""

import time
import socket
import numpy as np
import threading
from typing import Optional

from ._rhx_config import RHXConfig

FRAMES_PER_BLOCK = 128
MAGIC_NUMBER = 0x2ef07a08


class IntanRHXDevice(RHXConfig):
    """
    Class for interacting with the Intan RHX system over TCP/IP.

    Inherits:
        RHXConfig: Provides configuration and command utilities.

    Responsibilities:
        - Connects to command and data ports
        - Streams and parses EMG data
        - Records EMG data into memory

    Parameters:
        host (str): IP address of the RHX server.
        command_port (int): TCP port for command communication.
        data_port (int): TCP port for waveform data.
        num_channels (int): Number of channels to collect.
        sample_rate (float): Expected EMG sample rate.
        verbose (bool): Enable debug logging.
    """
    def __init__(self,
                 host="127.0.0.1",
                 command_port=5000,
                 data_port=5001,
                 num_channels=128,
                 sample_rate=None,
                 buffer_duration_sec=5,
                 auto_start=False,
                 verbose=False):
        self.host = host
        self.command_port = command_port
        self.data_port = data_port
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.verbose = verbose
        self.connected = False

        self.buffer_duration_sec = buffer_duration_sec
        self.circular_buffer = None
        self.circular_idx = 0
        self.buffer_lock = threading.Lock()
        self.streaming_thread = None
        self.streaming = False
        self.connected = False

        self.bytes_per_frame = 4 + 2 * self.num_channels
        self.bytes_per_block = 4 + FRAMES_PER_BLOCK * self.bytes_per_frame
        self.blocks_per_write = 1
        self.read_size = self.bytes_per_block * 1
        self._synced = False

        self.connect()

        if self.connected:
            if self.sample_rate is None:
                self.sample_rate = self.get_sample_rate()
            self._sample_counter = 0
            self.effective_fs = float(self.sample_rate)
            self.init_circular_buffer()
            super().__init__(self.command_socket, verbose=verbose)

        if auto_start:
            self.start_streaming()

    def connect(self):
        """Establish TCP connections to the RHX command and data ports."""
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

            self.connected = True
            self.connection_lost = False
            self.reconnect_attempts = 0

        except Exception as e:
            print("Failed to initialize connection with 'Remote TCP Control'")
            self.connected = False

    def receive_data(self, buffer: bytearray, read_size: int, max_reads: int = 16):
        """Try to drain multiple chunks from the socket without blocking forever."""
        for _ in range(max_reads):
            try:
                chunk = self.data_socket.recv(read_size)
                if not chunk:
                    break
                buffer.extend(chunk)
                if len(chunk) < read_size:
                    break
            except socket.timeout:
                break
        return buffer

    def init_circular_buffer(self):
        """Initialize the circular buffer for real-time data."""
        buffer_length = int(self.sample_rate * self.buffer_duration_sec)
        self.circular_buffer = np.zeros((self.num_channels, buffer_length), dtype=np.float32)
        self.circular_idx = 0

    def parse_emg_stream_fast(self, raw_bytes: bytearray, synced=True):
        """Fast EMG stream parser."""
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
        """Slow reference parser (for debugging). Prefer parse_emg_stream_fast."""
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
        """Background worker thread for streaming data."""
        rolling_buffer = bytearray()
        self.set_run_mode("run")

        try:
            while self.streaming:
                rolling_buffer = self.receive_data(rolling_buffer, self.read_size)
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

        finally:
            self.set_run_mode("stop")

    def _update_read_size(self):
        """Update read size based on channel count and blocks per write."""
        self.bytes_per_frame = 4 + 2 * self.num_channels
        self.bytes_per_block = 4 + FRAMES_PER_BLOCK * self.bytes_per_frame
        self.read_size = self.bytes_per_block * max(1, int(getattr(self, "blocks_per_write", 1)))

    def configure(self, **kwargs):
        """Configure device parameters."""
        for key, value in kwargs.items():
            if "blocks_per_write" in key:
                self.set_blocks_per_write(value)
                self.blocks_per_write = max(1, int(value))
                self._update_read_size()
            elif "enable_wide_channel" in key:
                self.enable_wide_channel(value)
                self._update_read_size()

    def start_streaming(self):
        """Start streaming data from the RHX device."""
        if self.streaming:
            print("Already streaming")
            return

        if self.sample_rate is None:
            self.sample_rate = self.get_sample_rate()
        self.effective_fs = float(self.sample_rate)
        self.init_circular_buffer()

        try:
            self.set_blocks_per_write(1)
            self.blocks_per_write = 1
            self._update_read_size()
        except Exception:
            pass

        self.streaming = True
        self.streaming_thread = threading.Thread(target=self._streaming_worker, daemon=True)
        self.streaming_thread.start()

    def stop_streaming(self):
        """Stop streaming data from the RHX device."""
        self.streaming = False
        if self.streaming_thread is not None:
            self.streaming_thread.join()
            self.streaming_thread = None

    def get_latest_window(self, duration_ms=200):
        """Get the latest data window from the circular buffer."""
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
        """
        Return the latest window plus the current circular buffer cursor.

        The returned cursor can be compared between successive calls to determine
        whether new samples have arrived.
        """
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
        """Record EMG data for a specified duration."""
        total_samples = int(self.sample_rate * duration_sec)
        collected_emg = np.zeros((self.num_channels, total_samples), dtype=np.float32)
        write_index = 0
        rolling_buffer = bytearray()
        sample_counter = 0
        last_print = time.time()

        self.set_run_mode("run")
        if verbose:
            print(f"[-->] Recording {duration_sec}s of EMG data...")

        try:
            while write_index < total_samples:
                rolling_buffer = self.receive_data(rolling_buffer, self.read_size)
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
                    print(f"[Rate] {rate:.2f} samples/sec")
                    last_print = now
                    sample_counter = 0

        finally:
            self.set_run_mode("stop")
            return collected_emg

    def close(self, stop_after_disconnect=True):
        """Close the TCP connections."""
        if stop_after_disconnect:
            if self.get_run_mode() == 'run':
                self.set_run_mode("stop")
                if self.verbose:
                    print("Runmode set to stop before closing.")

        self.command_socket.close()
        self.data_socket.close()

    def record_to_file(self, path, duration_sec=10):
        """Record EMG data to a file."""
        emg = self.record(duration_sec)
        np.savez(path, emg=emg, sample_rate=self.sample_rate)
        if self.verbose:
            print(f"Saved EMG to: {path}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()