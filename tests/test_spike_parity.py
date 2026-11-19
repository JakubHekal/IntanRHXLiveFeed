import unittest

import numpy as np

from leech.device.intan_rhx._processing_tasks import _run_all_channels


def _golden_run(fs=20000.0, n_ch=128):
    rng = np.random.default_rng(20261119)
    n = 24000
    t = np.arange(n, dtype=np.float64) / fs
    y = rng.standard_normal((n_ch, n)).astype(np.float64) * 8.0
    for k in range(n_ch):
        for p in (1000 + 37 * k, 7000 + 53 * k, 14000 + 41 * k,
                  19000 + 71 * k):
            if p + 30 < n:
                y[k, p:p + 20] += 400.0 * (1 if k % 2 == 0 else -1)

    last_scans = {}
    hist_states = {}
    caches = {}
    results = []
    for start in (0, 8000, 16000):
        stop = min(n, start + 8000)
        res = _run_all_channels(
            y[:, start:stop], t[start:stop], fs, 17,
            caches, last_scans, False, True, hist_states,
            abs_start=start, spike_bin_sec=5.0,
        )
        caches = res.all_spike_caches
        last_scans = res.last_scans
        hist_states = res.hist_states
        results.append(res)
    return results


class SpikeParityTest(unittest.TestCase):
    """Pins current spike behaviour so the incremental rewrite must match exactly."""

    def test_golden_run_is_deterministic(self):
        a = _golden_run()
        b = _golden_run()
        for ra, rb in zip(a, b):
            self.assertEqual(ra.last_scans, rb.last_scans)
            self.assertEqual(ra.all_spike_caches, rb.all_spike_caches)
            for ch in ra.hist_states:
                np.testing.assert_array_equal(
                    ra.hist_states[ch]['counts'], rb.hist_states[ch]['counts']
                )

    def test_golden_counts_are_stable(self):
        results = _golden_run()
        counts = [len(res.all_spike_caches[ch]) for res in results for ch in (0, 17, 127)]
        self.assertEqual(counts, [2, 2, 1, 3, 3, 2, 4, 4, 3])
        final_counts = results[-1].hist_states[17]['counts']
        self.assertEqual(int(final_counts.sum()), 4)

    def test_carried_filter_state_matches_golden(self):
        fs = 20000.0
        n = 24000
        rng = np.random.default_rng(20261119)
        t = np.arange(n, dtype=np.float64) / fs
        y = rng.standard_normal((128, n)).astype(np.float64) * 8.0
        for k in range(128):
            for p in (1000 + 37 * k, 7000 + 53 * k, 14000 + 41 * k,
                      19000 + 71 * k):
                if p + 30 < n:
                    y[k, p:p + 20] += 400.0 * (1 if k % 2 == 0 else -1)

        caches = {}
        last_scans = {}
        hist_states = {}
        state = None
        results = []
        for start in (0, 8000, 16000):
            stop = min(n, start + 8000)
            res = _run_all_channels(
                y[:, start:stop], t[start:stop], fs, 17,
                caches, last_scans, False, True, hist_states,
                abs_start=start, spike_bin_sec=5.0, filter_state=state,
            )
            caches = res.all_spike_caches
            last_scans = res.last_scans
            hist_states = res.hist_states
            state = res.filter_state
            results.append(res)

        counts = [len(res.all_spike_caches[ch]) for res in results for ch in (0, 17, 127)]
        self.assertEqual(counts, [2, 2, 1, 3, 3, 2, 4, 4, 3])
        self.assertEqual(state.length, 24000)
        self.assertLessEqual(state.length, state.cap)

    def test_filter_state_resets_after_gap(self):
        fs = 20000.0
        n = 24000
        rng = np.random.default_rng(1)
        t = np.arange(n, dtype=np.float64) / fs
        y = rng.standard_normal((128, n)).astype(np.float64) * 8.0

        res = _run_all_channels(
            y[:, :8000], t[:8000], fs, 0, {}, {}, False, True, {},
            abs_start=0, filter_state=None,
        )
        state = res.filter_state
        self.assertEqual(state.end_abs, 8000)

        # Skip 8000..16000: the stream worker dropped that data.
        res2 = _run_all_channels(
            y[:, 16000:], t[16000:], fs, 0,
            res.all_spike_caches, res.last_scans, False, True, res.hist_states,
            abs_start=16000, filter_state=state,
        )
        self.assertEqual(res2.filter_state.end_abs, 24000)
        self.assertEqual(res2.filter_state.length, 8000)


if __name__ == "__main__":
    unittest.main()
