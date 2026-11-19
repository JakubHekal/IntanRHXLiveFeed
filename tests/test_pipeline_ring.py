import unittest

import numpy as np

from leech.gui.ring_buffer import MinMaxModel, RingBuffer


class PipelineRingTest(unittest.TestCase):
    def test_dtype_split_and_wrap_continuity(self):
        rb = RingBuffer(1000.0, 3, duration_sec=0.1)
        self.assertEqual(rb.cap, 100)
        self.assertEqual(rb._y.dtype, np.float32)
        self.assertEqual(rb._t.dtype, np.float64)

        t = np.arange(250, dtype=np.float64) / 1000.0
        y = np.stack([t, -t, np.full(250, 7.0)])
        rb.write(t[:150], y[:, :150])
        rb.write(t[150:], y[:, 150:])
        t_read, y_read = rb.read_channel(0)
        np.testing.assert_array_equal(t_read, t[150:])
        np.testing.assert_allclose(y_read, t[150:], atol=1e-6)
        self.assertEqual(rb.raw_time_bounds(), (0.15, 0.249))
        self.assertEqual(y_read.dtype, np.float32)

    def test_model_matches_direct_minmax_and_caches(self):
        sr = 10000.0
        rng = np.random.default_rng(4)
        t = np.arange(5000, dtype=np.float64) / sr
        y = rng.standard_normal((1, t.size)).astype(np.float32)
        ring = RingBuffer(sr, 1, duration_sec=0.5)
        ring.write(t[:3000], y[:, :3000])
        model = MinMaxModel(sr, duration_sec=1.0)
        model.append_from_ring(ring, 0)
        v1 = model.version
        x, env = model.read(0.0, 0.15)
        self.assertIs(x, model.read(0.0, 0.15)[0])
        expected = y[0, t < 0.15]
        np.testing.assert_allclose([env.min(), env.max()],
                                   [expected.min(), expected.max()], atol=1e-6)
        ring.write(t[3000:], y[:, 3000:])
        model.append_from_ring(ring, 0)
        self.assertGreater(model.version, v1)
        x, env = model.read(0.0, 0.5)
        expected = y[0, t < 0.5]
        np.testing.assert_allclose([env.min(), env.max()],
                                   [expected.min(), expected.max()], atol=1e-6)

    def test_model_merges_partial_buckets(self):
        sr = 10000.0
        ring = RingBuffer(sr, 1, duration_sec=0.1)
        model = MinMaxModel(sr, duration_sec=1.0)
        t1 = np.array([0.0001, 0.0002], dtype=np.float64)
        ring.write(t1, np.array([[1.0, 5.0]]))
        model.append_from_ring(ring, 0)
        t2 = np.array([0.0003, 0.0004], dtype=np.float64)
        ring.write(t2, np.array([[-3.0, 2.0]]))
        model.append_from_ring(ring, 0)
        _, env = model.read(0.0, 0.01)
        np.testing.assert_allclose([env.min(), env.max()], [-3.0, 5.0])


if __name__ == "__main__":
    unittest.main()
