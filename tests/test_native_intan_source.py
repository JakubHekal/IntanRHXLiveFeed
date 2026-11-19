import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from leech.device.intan_rhx.replay import IntanNativeSource
from leech.workers.replay_source import make_source

RHD_MAGIC = 0xC6912702
RHS_MAGIC = 0xD69127AC
UV_PER_COUNT = np.float32(0.195)

REPO = Path(__file__).resolve().parents[1]
RHD_FIXTURE = (
    REPO / "experiments/Stimulation/runs/20261006_133722/raw"
    / "20ca1f42-f237-5248-a5de-6400f4fd4f9a"
)
RHS_FIXTURE = (
    REPO / "experiments/Long miniSMU/runs/PT 300nA/raw"
    / "Intan_RHX_66a9ea56-d59a-47c0-b781-8e9deb93b7ad"
)


def _q(value):
    if value is None:
        return struct.pack("<I", 0xFFFFFFFF)
    data = value.encode("utf-16-le")
    return struct.pack("<I", len(data)) + data


def _channel(native, custom, signal_type, enabled, order, *, rhs):
    record = _q(native) + _q(custom)
    record += struct.pack("<hhhhh", order, order, signal_type, enabled, order)
    if rhs:
        record += struct.pack("<h", 0)  # command stream
    record += struct.pack("<h", 0)  # board stream
    record += struct.pack("<hhhh", 1, -70, 0, 0)  # trigger settings
    record += struct.pack("<ff", 0.0, 0.0)  # impedance
    return record


def _group(channels, *, rhs):
    amps = sum(1 for channel in channels if channel[2] == 0)
    record = _q("Port A") + _q("A") + struct.pack("<hhh", 1, len(channels), amps)
    for order, (native, custom, signal_type, enabled) in enumerate(channels):
        record += _channel(native, custom, signal_type, enabled, order, rhs=rhs)
    return record


def _rhd_header(sample_rate, channels, *, temp_sensors=0, version=(3, 5)):
    main, secondary = version
    data = struct.pack("<Ihh", RHD_MAGIC, main, secondary)
    data += struct.pack("<f", sample_rate)
    data += struct.pack("<h", 1)  # DSP enabled
    data += struct.pack("<ffffff", 0.1, 0.1, 7500.0, 0.1, 0.1, 7500.0)
    data += struct.pack("<hff", 0, 1000.0, 1000.0)
    data += _q("") + _q("") + _q("")
    if main + secondary / 10.0 > 1.09:
        data += struct.pack("<h", temp_sensors)
    if main + secondary / 10.0 > 1.29:
        data += struct.pack("<h", 0)
    if main + secondary / 10.0 > 1.99:
        data += _q("n/a")
    data += struct.pack("<h", 1)
    return data + _group(channels, rhs=False)


def _rhs_header(sample_rate, channels, *, dc_saved):
    data = struct.pack("<Ihh", RHS_MAGIC, 3, 5)
    data += struct.pack("<f", sample_rate)
    data += struct.pack("<h", 1)  # DSP enabled
    data += struct.pack("<ffff", 0.1, 0.1, 0.1, 7500.0)
    data += struct.pack("<ffff", 0.1, 0.1, 0.1, 7500.0)
    data += struct.pack("<hff", 0, 1000.0, 1000.0)
    data += struct.pack("<hhfff", 0, 0, 10.0, 0.0, 0.0)
    data += _q("") + _q("") + _q("")
    data += struct.pack("<hh", int(dc_saved), 0)
    data += _q("n/a")
    data += struct.pack("<h", 1)
    return data + _group(channels, rhs=True)


def _blocks(samples, amps, *, dc=None, stim=None, extras=None):
    """Serialize amplifier blocks in Intan's channel-major block layout."""
    extras = extras or {}
    amp_blocks = np.asarray(amps, dtype="<u2")
    block_count = amp_blocks.shape[1] // samples
    parts = []
    for index in range(block_count):
        start = index * samples
        stop = start + samples
        parts.append(np.arange(samples, dtype="<i4").tobytes())
        parts.append(amp_blocks[:, start:stop].tobytes())
        if dc is not None:
            parts.append(np.asarray(dc, dtype="<u2")[:, start:stop].tobytes())
        if stim is not None:
            parts.append(np.asarray(stim, dtype="<u2")[:, start:stop].tobytes())
        for name, dtype in (
            ("aux", "<u2"), ("supply", "<u2"), ("temp", "<i2"),
            ("adc", "<u2"), ("dac", "<u2"), ("digin", "<u2"), ("digout", "<u2"),
        ):
            value = extras.get(name)
            if value is None:
                continue
            value = np.asarray(value, dtype=dtype)
            if name == "aux":
                value = value[:, index * (samples // 4):(index + 1) * (samples // 4)]
            elif name in ("supply", "temp"):
                per_block = value.size // block_count
                value = value.reshape(-1)[index * per_block:(index + 1) * per_block]
            else:
                value = value[..., start:stop]
            parts.append(value.tobytes())
    return b"".join(parts)


def _expected(amps):
    return (np.asarray(amps, dtype=np.float32) - 32768.0) * UV_PER_COUNT


def _read_all(source, step=37):
    pieces = []
    while not source.exhausted:
        chunk, markers = source.read(step)
        assert markers == []
        if chunk is None:
            break
        pieces.append(chunk)
    source.close()
    return np.concatenate(pieces, axis=1)


class NativeIntanSourceTest(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._temp_dir.name)

    def tearDown(self):
        self._temp_dir.cleanup()

    def test_rhd_reads_header_labels_and_scaled_blocks(self):
        channels = [
            ("A-000", "C0", 0, 1),
            ("A-001", "C1", 0, 1),
            ("A-AUX1", "", 1, 1),
            ("A-VDD1", "", 2, 1),
            ("A-ADC1", "", 3, 1),
            ("A-DIN1", "", 4, 1),
            ("A-DOUT1", "", 5, 1),
        ]
        raw = np.arange(2 * 256, dtype=np.uint16).reshape(2, 256) + 30000
        extras = {
            "aux": np.full((1, 64), 7),
            "supply": np.full(2, 9),
            "temp": np.full(2, 25),
            "adc": np.full((1, 256), 11),
            "digin": np.full(256, 13),
            "digout": np.full(256, 17),
        }
        path = self.tmp_path / "000001_Test_000001.rhd"
        path.write_bytes(
            _rhd_header(20000.0, channels, temp_sensors=1)
            + _blocks(128, raw, extras=extras)
        )

        source = make_source(self.tmp_path, "intan")
        assert isinstance(source, IntanNativeSource)
        assert source.sample_rate == 20000.0
        assert source.num_channels == 2
        assert source.labels == ["C0", "C1"]
        assert source.total_samples == 256

        joined = _read_all(source)
        np.testing.assert_array_equal(joined, _expected(raw))

    def test_rhs_dc_and_stim_sections_do_not_shift_amp_data(self):
        channels = [("A-000", "", 0, 1), ("A-001", "custom", 0, 1)]
        raw = np.arange(2 * 256, dtype=np.uint16).reshape(2, 256) + 10000
        dc = np.full((2, 256), 60000, dtype="<u2")
        stim = np.full((2, 256), 30000, dtype="<u2")
        path = self.tmp_path / "000001_Test_000001.rhs"
        path.write_bytes(
            _rhs_header(20000.0, channels, dc_saved=True)
            + _blocks(128, raw, dc=dc, stim=stim)
        )

        source = make_source(self.tmp_path, "rhs")
        assert source.sample_rate == 20000.0
        assert source.num_channels == 2
        assert source.labels == ["A-000", "custom"]
        assert source.total_samples == 256
        np.testing.assert_array_equal(_read_all(source), _expected(raw))

    def test_rhs_without_dc_section(self):
        channels = [("A-000", "", 0, 1)]
        raw = np.arange(128, dtype=np.uint16).reshape(1, 128) + 20000
        path = self.tmp_path / "000001_Test_000001.rhs"
        path.write_bytes(
            _rhs_header(20000.0, channels, dc_saved=False)
            + _blocks(128, raw, stim=np.zeros((1, 128), dtype="<u2"))
        )
        source = make_source(self.tmp_path, "rhd")
        assert source.total_samples == 128
        np.testing.assert_array_equal(_read_all(source, step=10), _expected(raw))

    def test_split_files_concatenate_in_natural_path_order(self):
        first_dir = self.tmp_path / "000001_Baseline_261006_133723"
        second_dir = self.tmp_path / "000002_Post_261006_133845"
        first_dir.mkdir()
        second_dir.mkdir()
        channels = [("A-000", "", 0, 1), ("A-001", "", 0, 1)]
        first_raw = np.arange(2 * 128, dtype=np.uint16).reshape(2, 128) + 30000
        second_raw = np.arange(2 * 128, dtype=np.uint16).reshape(2, 128) + 40000
        (second_dir / "000002_Post_261006_133945.rhd").write_bytes(
            _rhd_header(20000.0, channels) + _blocks(128, second_raw)
        )
        (first_dir / "000001_Baseline_261006_133823.rhd").write_bytes(
            _rhd_header(20000.0, channels) + _blocks(128, first_raw)
        )

        source = make_source(self.tmp_path, "native")
        assert source.total_samples == 256
        joined = _read_all(source, step=50)
        np.testing.assert_array_equal(
            joined, np.concatenate((_expected(first_raw), _expected(second_raw)), axis=1)
        )

    def test_rhd_v1_uses_sixty_sample_blocks(self):
        channels = [("A-000", "", 0, 1)]
        raw = np.arange(60, dtype=np.uint16).reshape(1, 60) + 32000
        path = self.tmp_path / "000001_Old.rhd"
        path.write_bytes(
            _rhd_header(20000.0, channels, version=(1, 0)) + _blocks(60, raw)
        )
        source = make_source(self.tmp_path, "rhd")
        assert source.total_samples == 60
        np.testing.assert_array_equal(_read_all(source, step=17), _expected(raw))

    def test_truncated_file_names_the_file(self):
        channels = [("A-000", "", 0, 1)]
        raw = np.arange(128, dtype=np.uint16).reshape(1, 128)
        path = self.tmp_path / "000001_Truncated.rhd"
        path.write_bytes(_rhd_header(20000.0, channels) + _blocks(128, raw)[:-3])

        with self.assertRaises(ValueError) as error:
            make_source(self.tmp_path, "rhs")
        assert path.name in str(error.exception)
        assert "block size" in str(error.exception)

    def test_mismatched_channel_counts_raise(self):
        (self.tmp_path / "a").mkdir()
        (self.tmp_path / "b").mkdir()
        one = [("A-000", "", 0, 1)]
        two = [("A-000", "", 0, 1), ("A-001", "", 0, 1)]
        (self.tmp_path / "a" / "000001_A.rhd").write_bytes(
            _rhd_header(20000.0, one) + _blocks(128, np.zeros((1, 128), dtype=np.uint16))
        )
        (self.tmp_path / "b" / "000002_B.rhd").write_bytes(
            _rhd_header(20000.0, two) + _blocks(128, np.zeros((2, 128), dtype=np.uint16))
        )

        with self.assertRaises(ValueError) as error:
            make_source(self.tmp_path, "rhs")
        message = str(error.exception)
        assert "disagree" in message
        assert "000001_A.rhd" in message and "000002_B.rhd" in message

    def test_unknown_magic_raises(self):
        path = self.tmp_path / "000001_Bad.rhd"
        path.write_bytes(struct.pack("<I", 0x12345678) + b"\x00" * 64)
        with self.assertRaises(ValueError) as error:
            make_source(self.tmp_path, "rhs")
        assert path.name in str(error.exception)
        assert "magic" in str(error.exception)

    def test_unsupported_format_still_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported replay format"):
            make_source(self.tmp_path, "nope")

    @unittest.skipUnless(RHD_FIXTURE.exists(), "real RHD fixture not present")
    def test_real_rhd_fixture_first_ten_seconds(self):
        source = make_source(RHD_FIXTURE, "rhs", device_type="rhx")
        assert source.sample_rate == 20000.0
        assert source.num_channels == 64
        assert source.total_samples == 2819072
        assert source.labels[0] == "A-000"
        chunk, markers = source.read(10 * 20000)
        source.close()
        assert markers == []
        assert chunk.shape == (64, 200000)
        assert chunk.dtype == np.float32
        assert np.isfinite(chunk).all()

    @unittest.skipUnless(RHS_FIXTURE.exists(), "real RHS fixture not present")
    def test_real_rhs_fixture_header_and_first_two_seconds(self):
        source = make_source(RHS_FIXTURE, "rhs", device_type="rhx")
        assert source.sample_rate == 20000.0
        assert source.num_channels == 1
        assert source.labels == ["A-000"]
        assert source.total_samples > 0
        chunk, markers = source.read(2 * 20000)
        source.close()
        assert markers == []
        assert chunk.shape == (1, 40000)
        assert np.isfinite(chunk).all()


if __name__ == "__main__":
    unittest.main()
