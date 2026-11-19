"""Replay source factory.

Device-specific readers live with their device:

    device/intan_rhx/replay.py  -> IntanNativeSource
    device/mcs_mea/replay.py   -> McsHdf5Source
    device/csv_source.py       -> CsvSource (generic ChunkWriter layout)

Sources expose the same small interface used by ReplaySession:

    source.sample_rate        -> float
    source.num_channels       -> int
    source.labels             -> list[str]
    source.total_samples      -> int | None   (None when not known up front)
    source.read(n)            -> (chunk, markers) | (None, [])
    source.exhausted          -> bool

``chunk`` is a float32 matrix of shape (channels, samples) in microvolts.
``markers`` is a list of (offset_in_chunk, marker_id, marker_name).
"""

from __future__ import annotations

from pathlib import Path


def make_source(
    chunks_dir: Path,
    fmt: str,
    sample_rate: float = 20000.0,
    label_order=None,
    device_type=None,
):
    fmt = str(fmt or "").lower()
    if fmt in ("rhs", "rhd", "native", "intan"):
        from leech.device.intan_rhx.replay import IntanNativeSource

        return IntanNativeSource(
            chunks_dir,
            sample_rate=sample_rate,
            label_order=label_order,
            device_type=device_type,
        )
    if fmt == "hdf5":
        from leech.device.mcs_mea.replay import McsHdf5Source

        return McsHdf5Source(chunks_dir, label_order=label_order)
    if fmt == "csv":
        from leech.device.csv_source import CsvSource

        return CsvSource(chunks_dir, sample_rate=sample_rate, label_order=label_order)
    raise ValueError(f"Unsupported replay format: {fmt}")


def __getattr__(name):
    """Keep old imports working without eagerly loading device modules."""
    if name == "CsvSource":
        from leech.device.csv_source import CsvSource

        return CsvSource
    if name in ("Hdf5Source", "McsHdf5Source"):
        from leech.device.mcs_mea.replay import McsHdf5Source

        return McsHdf5Source
    if name == "chunk_files":
        from leech.device.replay_paths import chunk_files

        return chunk_files
    raise AttributeError(name)


__all__ = ["make_source", "CsvSource", "Hdf5Source", "McsHdf5Source", "chunk_files"]
