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
        # ponytail: direct ring read of n samples, O(n) not O(cap) — regressed into
        # read_channel call during RingBuffer extraction, restored from original O(n) impl.
        if cnt >= self.cap:
            start = (self._wpos - n) % self.cap
            tail = self.cap - start
            if tail >= n:
                return (self._ring[0, start:start + n].copy(),
                        self._ring[1 + ch_idx, start:start + n].copy())
            t = np.empty(n, dtype=np.float64)
            y = np.empty(n, dtype=np.float64)
            t[:tail] = self._ring[0, start:]
            y[:tail] = self._ring[1 + ch_idx, start:]
            head = n - tail
            t[tail:] = self._ring[0, :head]
            y[tail:] = self._ring[1 + ch_idx, :head]
            return t, y
        start = cnt - n
        return self._ring[0, start:cnt].copy(), self._ring[1 + ch_idx, start:cnt].copy()

    def read_tail_matrix(self, n: int):
        cnt = min(self._total, self.cap)
        n = min(n, cnt)
        if n == 0:
            return np.array([]), np.empty((0, 0))
        if cnt >= self.cap:
            start = (self._wpos - n) % self.cap
            tail = self.cap - start
            if tail >= n:
                return (self._ring[0, start:start + n].copy(),
                        self._ring[1:, start:start + n].copy())
            t = np.empty(n, dtype=np.float64)
            y = np.empty((self.num_channels, n), dtype=np.float64)
            t[:tail] = self._ring[0, start:]
            y[:, :tail] = self._ring[1:, start:]
            head = n - tail
            t[tail:] = self._ring[0, :head]
            y[:, tail:] = self._ring[1:, :head]
            return t, y
        start = cnt - n
        return self._ring[0, start:cnt].copy(), self._ring[1:, start:cnt].copy()

    def raw_time_bounds(self):
        if self._total == 0:
            return 0.0, 0.0
        # ponytail: peek at first/last timestamps directly, O(1) — restored from
        # original impl that computed arithmetically; read_channel(0) copy was a regression.
        cnt = min(self._total, self.cap)
        if cnt >= self.cap:
            return (float(self._ring[0, self._wpos]),
                    float(self._ring[0, (self._wpos - 1) % self.cap]))
        return float(self._ring[0, 0]), float(self._ring[0, cnt - 1])

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


if __name__ == "__main__":

    def _check():
        import timeit

        rb = RingBuffer(sample_rate=20000, num_channels=2, duration_sec=1)
        cap = rb.cap
        sr = rb.sample_rate

        # empty
        assert rb.raw_time_bounds() == (0.0, 0.0)
        t, y = rb.read_tail(100)
        assert t.size == 0 and y.size == 0

        # partial fill (no wrap)
        n = 5000
        t_w = np.arange(n, dtype=np.float64) / sr
        y_w = np.ones((2, n), dtype=np.float64) * 42.0
        y_w[1, :] = 99.0
        rb.write(t_w, y_w)
        assert rb._total == n

        lo, hi = rb.raw_time_bounds()
        assert abs(lo - 0.0) < 1e-12, lo
        assert abs(hi - (n - 1) / sr) < 1e-12, hi

        # read_tail vs read_channel ground truth
        for ch in (0, 1):
            t_full, y_full = rb.read_channel(ch)
            for tail_n in (1, 100, 4999, 5000):
                t_tail, y_tail = rb.read_tail(tail_n, ch_idx=ch)
                np.testing.assert_array_equal(t_tail, t_full[-tail_n:])
                np.testing.assert_array_equal(y_tail, y_full[-tail_n:])

        # fill to capacity (wrap)
        remaining = cap - n
        t_w2 = np.arange(n, n + remaining, dtype=np.float64) / sr
        y_w2 = np.ones((2, remaining), dtype=np.float64) * 7.0
        y_w2[1, :] = 3.0
        rb.write(t_w2, y_w2)
        assert rb._total == cap

        lo, hi = rb.raw_time_bounds()
        # oldest = first written that survived = index 0, newest = index cap-1
        assert abs(lo - 0.0) < 1e-12, lo
        assert abs(hi - (cap - 1) / sr) < 1e-12, hi

        # now overflow to trigger wrap-around in write
        overflow_n = 3000
        t_w3 = np.arange(cap, cap + overflow_n, dtype=np.float64) / sr
        y_w3 = np.ones((2, overflow_n), dtype=np.float64) * 11.0
        y_w3[1, :] = 22.0
        rb.write(t_w3, y_w3)
        assert rb._total == cap + overflow_n
        # oldest = t at wpos, newest = t at (wpos-1)%cap
        lo, hi = rb.raw_time_bounds()
        assert abs(lo - (overflow_n) / sr) < 1e-12, lo
        assert abs(hi - (cap + overflow_n - 1) / sr) < 1e-12, hi

        for ch in (0, 1):
            t_full, y_full = rb.read_channel(ch)
            for tail_n in (1, 100, 2000, cap):
                t_tail, y_tail = rb.read_tail(tail_n, ch_idx=ch)
                np.testing.assert_array_equal(t_tail, t_full[-tail_n:])
                np.testing.assert_array_equal(y_tail, y_full[-tail_n:])

        # perf: read_tail on full 6M-sample buffer
        rb_big = RingBuffer(sample_rate=20000, num_channels=1, duration_sec=300)
        t_fill = np.arange(rb_big.cap, dtype=np.float64) / 20000.0
        y_fill = np.random.default_rng(0).standard_normal((1, rb_big.cap))
        rb_big.write(t_fill, y_fill)
        n_vis = int(20000 * 10)
        elapsed = timeit.timeit(lambda: rb_big.read_tail(n_vis, ch_idx=0), number=50)
        avg_ms = elapsed / 50 * 1000
        print(f"read_tail({n_vis}) on full {rb_big.cap}-sample buffer: {avg_ms:.2f} ms avg")

        elapsed2 = timeit.timeit(lambda: rb_big.raw_time_bounds(), number=50)
        avg_ms2 = elapsed2 / 50 * 1000
        print(f"raw_time_bounds() on full {rb_big.cap}-sample buffer: {avg_ms2:.4f} ms avg")
        print("OK")

    _check()
