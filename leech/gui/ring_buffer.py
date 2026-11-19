import numpy as np

from leech.gui.plot_helpers import DISPLAY_BUFFER_SEC, MAX_DISPLAY_POINTS


MODEL_BUCKET_HZ = 1000.0
MODEL_BUFFER_SEC = DISPLAY_BUFFER_SEC


class RingBuffer:
    """Fixed-size per-channel ring. Samples float32, timestamps float64."""

    def __init__(self, sample_rate: float, num_channels: int, duration_sec: float = DISPLAY_BUFFER_SEC):
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.cap = max(100, int(round(sample_rate * duration_sec)))
        self._t = np.empty(self.cap, dtype=np.float64)
        self._y = np.empty((num_channels, self.cap), dtype=np.float32)
        self._wpos = 0
        self._total = 0
        self.version = 0

    def _put(self, pos: int, t: np.ndarray, y: np.ndarray):
        tail = self.cap - pos
        n = t.size
        if n <= tail:
            self._t[pos:pos + n] = t
            self._y[:, pos:pos + n] = y
        else:
            self._t[pos:] = t[:tail]
            self._y[:, pos:] = y[:, :tail]
            rem = n - tail
            self._t[:rem] = t[tail:]
            self._y[:, :rem] = y[:, rem:]

    def _gather(self, pos: int, n: int):
        tail = self.cap - pos
        if n <= tail:
            return self._t[pos:pos + n].copy(), self._y[:, pos:pos + n].copy()
        t = np.empty(n, dtype=np.float64)
        y = np.empty((self.num_channels, n), dtype=np.float32)
        t[:tail] = self._t[pos:]
        y[:, :tail] = self._y[:, pos:]
        rem = n - tail
        t[tail:] = self._t[:rem]
        y[:, tail:] = self._y[:, :rem]
        return t, y

    def write(self, t: np.ndarray, y: np.ndarray):
        t = np.asarray(t, dtype=np.float64).ravel()
        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(1, -1)
        n = min(t.size, y.shape[1])
        if n <= 0:
            return
        y = y.astype(np.float32, copy=False)
        if n >= self.cap:
            self._t[:] = t[-self.cap:]
            self._y[:] = y[:, -self.cap:]
            self._wpos = 0
            self._total += n
            self.version += 1
            return
        space = self.cap - self._wpos
        if n <= space:
            self._put(self._wpos, t[:n], y[:, :n])
        else:
            self._put(self._wpos, t[:space], y[:, :space])
            self._put(0, t[space:n], y[:, space:n])
        self._wpos = (self._wpos + n) % self.cap
        self._total += n
        self.version += 1

    def read_range(self, start: int, stop: int, ch_idx=None):
        """Read absolute sample indices [start, stop). ch_idx=None yields (ch, n)."""
        oldest = max(0, self._total - self.cap)
        start = max(oldest, min(int(start), self._total))
        stop = max(start, min(int(stop), self._total))
        n = stop - start
        if n == 0:
            if ch_idx is None:
                return np.array([], dtype=np.float64), np.empty((self.num_channels, 0), dtype=np.float32)
            return np.array([], dtype=np.float64), np.array([], dtype=np.float32)
        if self._total <= self.cap:
            pos = start
        else:
            pos = (self._wpos + start - oldest) % self.cap
        t, y = self._gather(pos, n)
        if ch_idx is not None:
            return t, y[int(ch_idx)]
        return t, y

    def read_channel(self, ch_idx: int = 0):
        return self.read_range(max(0, self._total - self.cap), self._total, ch_idx)

    def read_tail(self, n: int, ch_idx: int = 0):
        return self.read_range(max(0, self._total - n), self._total, ch_idx)

    def read_tail_matrix(self, n: int):
        return self.read_range(max(0, self._total - n), self._total, None)

    def raw_time_bounds(self):
        if self._total == 0:
            return 0.0, 0.0
        if self._total >= self.cap:
            return float(self._t[self._wpos]), float(self._t[(self._wpos - 1) % self.cap])
        return float(self._t[0]), float(self._t[self._total - 1])

    def resize(self, num_channels: int):
        self.num_channels = num_channels
        self._y = np.empty((num_channels, self.cap), dtype=np.float32)
        self._t = np.empty(self.cap, dtype=np.float64)
        self._wpos = 0
        self._total = 0
        self.version += 1

    def clear(self):
        self._wpos = 0
        self._total = 0
        self.version += 1

    @property
    def total(self):
        return self._total


class MinMaxModel:
    """1 kHz min/max envelope of the last 300 s, independent of raw ring length."""

    def __init__(self, sample_rate: float, duration_sec: float = MODEL_BUFFER_SEC,
                 bucket_hz: float = MODEL_BUCKET_HZ, num_channels: int = 1):
        self.sample_rate = sample_rate
        self.bucket_hz = bucket_hz
        self.num_channels = num_channels
        self.cap = max(100, int(round(duration_sec * bucket_hz)))
        self._x = np.empty(self.cap, dtype=np.float64)
        self._min = np.empty((num_channels, self.cap), dtype=np.float32)
        self._max = np.empty((num_channels, self.cap), dtype=np.float32)
        self.reset()

    def reset(self):
        self._wpos = 0
        self._total = 0
        self._last_bucket = -1
        self._cursor = 0
        self.version = 0
        self._cache = None
        self._cache_key = None

    def resize(self, num_channels: int):
        if num_channels == self.num_channels:
            return
        self.num_channels = num_channels
        self._min = np.empty((num_channels, self.cap), dtype=np.float32)
        self._max = np.empty((num_channels, self.cap), dtype=np.float32)
        self.reset()

    def append_from_ring(self, ring: 'RingBuffer', ch_idx=None):
        if ring.total < self._cursor or ring.num_channels != self.num_channels:
            self.reset()
        start = max(self._cursor, max(0, ring.total - ring.cap))
        if ring.total <= start:
            return
        t, y = ring.read_range(start, ring.total, ch_idx)
        self._cursor = ring.total
        if t.size == 0:
            return
        if y.ndim == 1:
            y = y.reshape(1, -1)
        buckets = np.floor(t * self.bucket_hz).astype(np.int64)
        buckets = np.clip(buckets, 0, None)
        order = np.argsort(buckets, kind='stable')
        b = buckets[order]
        v = y[:, order]
        uniq, starts = np.unique(b, return_index=True)
        if uniq.size == 0:
            return
        if self._total > 0 and uniq[0] > self._last_bucket + 1:
            self.reset()
            self._cursor = ring.total
        if uniq[0] < self._last_bucket or uniq[0] >= self._last_bucket + self.cap:
            self.reset()
            self._cursor = ring.total
        fresh = uniq > self._last_bucket
        pos = uniq % self.cap
        if np.any(fresh):
            self._min[:, pos[fresh]] = np.inf
            self._max[:, pos[fresh]] = -np.inf
            self._x[pos[fresh]] = (uniq[fresh] + 0.5) / self.bucket_hz
        chs = range(y.shape[0]) if ch_idx is None else [int(ch_idx)]
        for ch in chs:
            np.minimum.at(self._min[ch], pos, np.minimum.reduceat(v[ch], starts))
            np.maximum.at(self._max[ch], pos, np.maximum.reduceat(v[ch], starts))
        self._last_bucket = int(uniq[-1])
        self._total = max(self._total, self._last_bucket + 1)
        self._wpos = (self._last_bucket + 1) % self.cap
        self.version += 1

    def _gather(self, first: int, last: int, ch_idx: int = 0):
        n = last - first + 1
        oldest = max(0, self._total - self.cap)
        oldest_pos = (self._wpos - min(self._total, self.cap)) % self.cap
        pos = (oldest_pos + first - oldest) % self.cap
        tail = self.cap - pos
        ch = int(ch_idx)
        x = np.empty(n, dtype=np.float64)
        mn = np.empty(n, dtype=np.float32)
        mx = np.empty(n, dtype=np.float32)
        if n <= tail:
            x[:] = self._x[pos:pos + n]
            mn[:] = self._min[ch, pos:pos + n]
            mx[:] = self._max[ch, pos:pos + n]
        else:
            x[:tail] = self._x[pos:]
            mn[:tail] = self._min[ch, pos:]
            mx[:tail] = self._max[ch, pos:]
            x[tail:] = self._x[:n - tail]
            mn[tail:] = self._min[ch, :n - tail]
            mx[tail:] = self._max[ch, :n - tail]
        return x, mn, mx

    def read(self, x0: float, x1: float, max_points: int = MAX_DISPLAY_POINTS,
             ch_idx: int = 0):
        if self._total == 0 or x1 <= x0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float32)
        key = (float(x0), float(x1), int(max_points), int(ch_idx), self.version)
        if self._cache_key == key:
            return self._cache
        oldest = max(0, self._total - self.cap)
        first = max(oldest, int(np.floor(x0 * self.bucket_hz)))
        last = min(self._total - 1, int(np.ceil(x1 * self.bucket_hz)))
        if last < first:
            self._cache_key = key
            self._cache = (np.array([], dtype=np.float64), np.array([], dtype=np.float32))
            return self._cache
        x, mn, mx = self._gather(first, last, ch_idx)
        step = max(1, int(np.ceil(x.size / max(1, max_points // 2))))
        if step > 1:
            groups = int(np.ceil(x.size / step))
            pad = (-x.size) % step
            mn = np.pad(mn, (0, pad), mode='constant', constant_values=np.inf)
            mx = np.pad(mx, (0, pad), mode='constant', constant_values=-np.inf)
            starts = np.arange(0, groups * step, step)
            mn = np.minimum.reduceat(mn, starts)
            mx = np.maximum.reduceat(mx, starts)
            x_left = (first + starts) / self.bucket_hz
            x_right = np.minimum(first + starts + step, last + 1) / self.bucket_hz
        else:
            x_left = np.arange(first, last + 1, dtype=np.float64) / self.bucket_hz
            x_right = (np.arange(first, last + 1, dtype=np.float64) + 1.0) / self.bucket_hz
        x_out = np.empty(x_left.size * 2, dtype=np.float64)
        y_out = np.empty(x_left.size * 2, dtype=np.float32)
        x_out[0::2] = x_left
        x_out[1::2] = x_right
        y_out[0::2] = mn
        y_out[1::2] = mx
        self._cache_key = key
        self._cache = (x_out, y_out)
        return self._cache

    @property
    def total(self):
        return self._total


if __name__ == "__main__":
    def _check():
        import timeit

        sr = 20000
        rb = RingBuffer(sr, 2, duration_sec=1)
        cap = rb.cap
        assert rb.raw_time_bounds() == (0.0, 0.0)
        t, y = rb.read_tail(100)
        assert t.size == 0 and y.size == 0

        n = 5000
        t_w = np.arange(n, dtype=np.float64) / sr
        y_w = np.ones((2, n), dtype=np.float64) * 42.0
        y_w[1] = 99.0
        rb.write(t_w, y_w)
        assert rb._y.dtype == np.float32 and rb._t.dtype == np.float64
        assert rb._y.shape == (2, cap)
        assert rb._total == n
        lo, hi = rb.raw_time_bounds()
        assert abs(lo) < 1e-12 and abs(hi - (n - 1) / sr) < 1e-12

        for ch in (0, 1):
            t_full, y_full = rb.read_channel(ch)
            for tail_n in (1, 100, 4999, 5000):
                t_tail, y_tail = rb.read_tail(tail_n, ch_idx=ch)
                np.testing.assert_array_equal(t_tail, t_full[-tail_n:])
                np.testing.assert_array_equal(y_tail, y_full[-tail_n:])

        remaining = cap - n
        rb.write(np.arange(n, n + remaining, dtype=np.float64) / sr,
                 np.full((2, remaining), 7.0))
        overflow = 3000
        rb.write(np.arange(cap, cap + overflow, dtype=np.float64) / sr,
                 np.full((2, overflow), 11.0))
        lo, hi = rb.raw_time_bounds()
        assert abs(lo - overflow / sr) < 1e-12, lo
        assert abs(hi - (cap + overflow - 1) / sr) < 1e-12, hi
        for ch in (0, 1):
            t_full, y_full = rb.read_channel(ch)
            np.testing.assert_array_equal(t_full, np.arange(overflow, cap + overflow) / sr)
            for tail_n in (1, 100, 2000, cap):
                t_tail, y_tail = rb.read_tail(tail_n, ch_idx=ch)
                np.testing.assert_array_equal(t_tail, t_full[-tail_n:])
                np.testing.assert_array_equal(y_tail, y_full[-tail_n:])

        model = MinMaxModel(sr, duration_sec=2, bucket_hz=1000.0)
        rng = np.random.default_rng(0)
        model_t = np.arange(0, 15000, dtype=np.float64) / sr
        model_y = rng.standard_normal((1, model_t.size)).astype(np.float32)
        rb2 = RingBuffer(sr, 1, duration_sec=1)
        rb2.write(model_t, model_y)
        model.append_from_ring(rb2, 0)
        ex, ey = model.read(0.0, 0.75, 2500)
        assert ex.size > 0 and ey.size == ex.size
        direct = model_y[0, model_t < 0.751]
        assert abs(float(ey.min()) - float(direct.min())) < 1e-6
        assert abs(float(ey.max()) - float(direct.max())) < 1e-6
        v0 = model.version
        model.read(0.0, 0.5)
        assert model.version == v0
        model.reset()
        x, y = model.read(0.0, 0.75)
        assert x.size == 0 and y.size == 0

        rb_big = RingBuffer(sr, 1, duration_sec=300)
        t_fill = np.arange(rb_big.cap, dtype=np.float64) / sr
        y_fill = rng.standard_normal((1, rb_big.cap))
        rb_big.write(t_fill, y_fill)
        n_vis = int(sr * 10)
        elapsed = timeit.timeit(lambda: rb_big.read_tail(n_vis, ch_idx=0), number=50)
        print(f"read_tail({n_vis}): {elapsed / 50 * 1000:.2f} ms avg")

        m_big = MinMaxModel(sr)
        m_big.append_from_ring(rb_big, 0)
        elapsed = timeit.timeit(lambda: m_big.read(290.0, 300.0), number=50)
        print(f"model.read(10 s): {elapsed / 50 * 1000:.3f} ms avg")
        print("OK")

    _check()
