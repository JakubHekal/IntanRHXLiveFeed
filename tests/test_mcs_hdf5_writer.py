import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from leech.device.mcs_mea.layout import channel_names, mea_layout, mea_name
from leech.workers.hdf5_chunk_writer import Hdf5ChunkWriter


ADC_STEP_UV = 0.059605


def _writer(tmp: str, num_channels: int, chunk_max_sec: float = 300.0) -> Hdf5ChunkWriter:
    writer = Hdf5ChunkWriter(
        sample_rate=20000,
        num_channels=num_channels,
        channel_names=channel_names(num_channels),
        chunk_max_sec=chunk_max_sec,
        flush_interval_sec=0,
        adc_step_uv=ADC_STEP_UV,
        mea_name=mea_name(num_channels),
        mea_layout=mea_layout(num_channels),
    )
    writer.raw_chunks_dir = Path(tmp)
    writer.filename_prefix = "test"
    writer._open_new_chunk_locked()
    return writer


class McsHdf5WriterTest(unittest.TestCase):
    def test_60_channel_file_is_mcs_rawdata_v3(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = _writer(tmp, 60)
            raw = (np.arange(600, dtype=np.int32).reshape(60, 10) - 300).astype(np.float64)
            microvolts = (raw * ADC_STEP_UV).astype(np.float32)
            writer.append_data(
                microvolts,
                marker_info={2: {"id": 7, "name": "stim"}},
                force_flush=True,
            )
            writer.close()

            path = next(Path(tmp).glob("chunk_test_000000.h5"))
            with h5py.File(path, "r") as f:
                self.assertEqual(f.attrs["McsHdf5ProtocolType"], b"RawData")
                self.assertEqual(f.attrs["McsHdf5ProtocolVersion"], 3)
                self.assertEqual(f.attrs["GeneratingApplicationName"], b"LEECH")

                data = f["Data"]
                self.assertEqual(data.attrs["MeaName"], b"60MEA200/30iR")
                self.assertEqual(data.attrs["MeaLayout"], b"60MEA200_30_iR")
                self.assertIn("FileGUID", data.attrs)

                recording = data["Recording_0"]
                self.assertEqual(recording.attrs["TimeStamp"], 0)
                self.assertEqual(recording.attrs["Duration"], 500)

                stream = recording["AnalogStream"]["Stream_0"]
                self.assertEqual(
                    set(stream.keys()),
                    {"InfoChannel", "ChannelData", "ChannelDataTimeStamps"},
                )
                info = stream["InfoChannel"]
                self.assertEqual(info.attrs["InfoVersion"], 2)
                self.assertEqual(
                    info.dtype.names,
                    (
                        "ChannelID", "RowIndex", "GroupID", "ElectrodeGroup", "Label",
                        "RawDataType", "Unit", "Exponent", "ADZero", "Tick",
                        "ConversionFactor", "ADCBits", "HighPassFilterType",
                        "HighPassFilterCutOffFrequency", "HighPassFilterOrder",
                        "LowPassFilterType", "LowPassFilterCutOffFrequency",
                        "LowPassFilterOrder",
                    ),
                )
                self.assertEqual(list(info["ChannelID"]), list(range(60)))
                self.assertEqual(list(info["RowIndex"]), list(range(60)))
                self.assertTrue(all(group == 0 for group in info["GroupID"]))
                self.assertTrue(all(group == 0 for group in info["ElectrodeGroup"]))
                labels = [value.decode() for value in info["Label"]]
                self.assertEqual(labels, channel_names(60))
                self.assertTrue(all(value == b"Int" for value in info["RawDataType"]))
                self.assertTrue(all(value == b"V" for value in info["Unit"]))
                self.assertTrue(all(value == -12 for value in info["Exponent"]))
                self.assertTrue(all(value == 50 for value in info["Tick"]))
                self.assertTrue(all(value == 59605 for value in info["ConversionFactor"]))
                self.assertTrue(all(value == 24 for value in info["ADCBits"]))

                channel_data = stream["ChannelData"]
                self.assertEqual(channel_data.shape, (60, 10))
                self.assertEqual(channel_data.dtype, np.dtype("int32"))
                self.assertEqual(channel_data.compression, "gzip")
                self.assertEqual(channel_data.compression_opts, 5)

                timestamps = stream["ChannelDataTimeStamps"]
                np.testing.assert_array_equal(timestamps[:], [[0, 0, 9]])
                self.assertEqual(timestamps.attrs["Column-0"], b"FirstTimeStamp")

                volts = channel_data[:].astype(np.float64) * 59605e-12
                np.testing.assert_allclose(volts, microvolts.astype(np.float64) * 1e-6, atol=1e-12)

                np.testing.assert_array_equal(f["markers/sample_index"][:], [2])
                np.testing.assert_array_equal(f["markers/marker_id"][:], [7])
                self.assertEqual([value.decode() for value in f["markers/marker_name"][:]], ["stim"])

    def test_rotation_keeps_timestamps_continuous(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = _writer(tmp, 60, chunk_max_sec=0.01)
            writer.append_data(np.zeros((60, 150), dtype=np.float32), force_flush=True)
            writer.append_data(np.zeros((60, 100), dtype=np.float32), force_flush=True)
            writer.close()

            paths = sorted(Path(tmp).glob("chunk_test_*.h5"))
            self.assertEqual(len(paths), 2)
            with h5py.File(paths[0], "r") as first:
                stream = first["Data/Recording_0/AnalogStream/Stream_0"]
                np.testing.assert_array_equal(stream["ChannelDataTimeStamps"][:], [[0, 0, 149]])
                self.assertEqual(first["Data/Recording_0"].attrs["Duration"], 7500)
            with h5py.File(paths[1], "r") as second:
                stream = second["Data/Recording_0/AnalogStream/Stream_0"]
                np.testing.assert_array_equal(stream["ChannelDataTimeStamps"][:], [[7500, 0, 99]])
                self.assertEqual(stream["ChannelData"].shape, (60, 100))
                self.assertEqual(second["Data/Recording_0"].attrs["TimeStamp"], 7500)
                self.assertEqual(second["Data/Recording_0"].attrs["Duration"], 12500)

    def test_120_channel_labels_match_verified_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = _writer(tmp, 120)
            writer.append_data(np.zeros((120, 4), dtype=np.float32), force_flush=True)
            writer.close()
            path = next(Path(tmp).glob("chunk_test_000000.h5"))
            with h5py.File(path, "r") as f:
                stream = f["Data/Recording_0/AnalogStream/Stream_0"]
                labels = [value.decode() for value in stream["InfoChannel"]["Label"]]
                self.assertEqual(labels, channel_names(120))
                self.assertEqual(f["Data"].attrs["MeaName"], b"120MEA200/30iR")
                self.assertEqual(f["Data"].attrs["MeaLayout"], b"120MEA200_30_iR")


if __name__ == "__main__":
    unittest.main()
