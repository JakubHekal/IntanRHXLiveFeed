import numpy as np


class RingBuffer:
    def __init__(self, sample_rate: float, num_channels: int, duration_sec: float = 300):
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.cap = max(100, int(round(sample_rate * duration_sec)))
        self._ring = np.zeros((1 + num_channels, self.cap), dtype=np.float64)
        self._wpos = 0
        self._total = 0

    def write(self, t: np.ndarray, y: np.ndarray):
        n = y.shape[1] if y.ndim == 2 else y.size
        if n == 0:
            return
        if y.ndim == 1:
            y = y.reshape(1, -1)
        if n >= self.cap:
            self._ring[0, :] = t[-self.cap:]
            self._ring[1:, :] = y[:, -self.cap:]
            self._wpos = 0
            self._total += n
        else:
            space = self.cap - self._wpos
            if n <= space:
                self._ring[0, self._wpos:self._wpos + n] = t
                self._ring[1:, self._wpos:self._wpos + n] = y
            else:
                self._ring[0, self._wpos:] = t[:space]
                self._ring[1:, self._wpos:] = y[:, :space]
                rem = n - space
                self._ring[0, :rem] = t[space:]
                self._ring[1:, :rem] = y[:, space:]
            self._wpos = (self._wpos + n) % self.cap
            self._total += n

    def read_channel(self, ch_idx: int = 0):
        if self._total == 0:
            return np.array([]), np.array([])
        cnt = min(self._total, self.cap)
        if cnt >= self.cap:
            idx = self._wpos
            t = np.empty(cnt, dtype=np.float64)
            y = np.empty(cnt, dtype=np.float64)
            tail = cnt - idx
            t[:tail] = self._ring[0, idx:]
            y[:tail] = self._ring[1 + ch_idx, idx:]
            t[tail:] = self._ring[0, :idx]
            y[tail:] = self._ring[1 + ch_idx, :idx]
            return t, y
        return self._ring[0, :cnt].copy(), self._ring[1 + ch_idx, :cnt].copy()

    def read_tail(self, n: int, ch_idx: int = 0):
        cnt = min(self._total, self.cap)
        n = min(n, cnt)
        if n == 0:
            return np.array([]), np.array([])
        t, y = self.read_channel(ch_idx)
        return t[-n:], y[-n:]

    def raw_time_bounds(self):
        if self._total == 0:
            return 0.0, 0.0
        t, _ = self.read_channel(0)
        if t.size == 0:
            return 0.0, 0.0
        return float(t[0]), float(t[-1])

    def resize(self, num_channels: int):
        self.num_channels = num_channels
        self._ring = np.zeros((1 + num_channels, self.cap), dtype=np.float64)
        self._wpos = 0
        self._total = 0

    def clear(self):
        self._ring[:] = 0
        self._wpos = 0
        self._total = 0

    @property
    def total(self):
        return self._total
