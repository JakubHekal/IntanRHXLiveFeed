"""Native Intan RHD/RHS header parsing and block reading."""

import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_RHD_MAGIC = 0xC6912702
_RHS_MAGIC = 0xD69127AC
_AMP_OFFSET = 32768.0
_AMP_SCALE_UV = 0.195
_NATIVE_SUFFIXES = {".rhd", ".rhs"}


def _natural_path_key(path: Path):
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", str(path))
    )


def native_files(chunks_dir: Path):
    """All Intan .rhd/.rhs files below chunks_dir, in natural path order."""
    root = Path(chunks_dir)
    if not root.exists():
        return []
    files = [
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in _NATIVE_SUFFIXES
    ]
    return sorted(files, key=_natural_path_key)


class _HeaderReader:
    def __init__(self, handle, path: Path):
        self._handle = handle
        self._path = path

    def _read(self, size: int) -> bytes:
        data = self._handle.read(size)
        if len(data) != size:
            raise ValueError(f"{self._path}: truncated Intan header")
        return data

    def u32(self) -> int:
        return struct.unpack("<I", self._read(4))[0]

    def i16(self) -> int:
        return struct.unpack("<h", self._read(2))[0]

    def f32(self) -> float:
        return struct.unpack("<f", self._read(4))[0]

    def qstring(self) -> str:
        length = self.u32()
        if length == 0xFFFFFFFF:
            return ""
        if length % 2 or length > 1 << 20:
            raise ValueError(f"{self._path}: invalid Intan string length {length}")
        return self._read(length).decode("utf-16-le")

    def tell(self) -> int:
        return self._handle.tell()


@dataclass
class IntanFile:
    path: Path
    sample_rate: float
    labels: list
    num_channels: int
    header_size: int
    samples_per_block: int
    num_blocks: int
    stride: int
    dtype: np.dtype

    @property
    def total_samples(self) -> int:
        return self.samples_per_block * self.num_blocks


def _block_dtype(samples: int, num_amp: int, is_rhs: bool, dc_saved: bool, counts, temp_count):
    fields = [
        ("timestamps", "<i4", (samples,)),
        ("amp", "<u2", (num_amp, samples)),
    ]
    if is_rhs:
        if dc_saved:
            fields.append(("dc", "<u2", (num_amp, samples)))
        fields.append(("stim", "<u2", (num_amp, samples)))
    if counts.get(1):
        fields.append(("aux", "<u2", (counts[1], samples // 4)))
    if counts.get(2):
        fields.append(("supply", "<u2", (counts[2],)))
    if temp_count:
        fields.append(("temp", "<i2", (temp_count,)))
    if counts.get(3):
        fields.append(("adc", "<u2", (counts[3], samples)))
    if is_rhs and counts.get(4):
        fields.append(("dac", "<u2", (counts[4], samples)))
    digin_type, digout_type = (5, 6) if is_rhs else (4, 5)
    if counts.get(digin_type):
        fields.append(("digin", "<u2", (samples,)))
    if counts.get(digout_type):
        fields.append(("digout", "<u2", (samples,)))
    return np.dtype(fields)


def read_intan_header(path: Path) -> IntanFile:
    with open(path, "rb") as handle:
        reader = _HeaderReader(handle, path)

        magic = reader.u32()
        if magic == _RHD_MAGIC:
            is_rhs = False
        elif magic == _RHS_MAGIC:
            is_rhs = True
        else:
            raise ValueError(f"{path}: unknown Intan magic 0x{magic:08X}")

        main_version = reader.i16()
        secondary_version = reader.i16()
        if main_version not in (1, 2, 3):
            raise ValueError(
                f"{path}: unsupported Intan header version "
                f"{main_version}.{secondary_version}"
            )
        version = main_version + secondary_version / 10.0

        sample_rate = reader.f32()
        if not 0.0 < sample_rate < 1.0e6:
            raise ValueError(f"{path}: invalid Intan sample rate {sample_rate}")

        reader.i16()  # DSP enabled
        reader.f32()  # actual DSP cutoff
        reader.f32()  # actual lower bandwidth
        if is_rhs:
            reader.f32()  # actual lower-settle bandwidth
        reader.f32()  # actual upper bandwidth
        reader.f32()  # desired DSP cutoff
        reader.f32()  # desired lower bandwidth
        if is_rhs:
            reader.f32()  # desired lower-settle bandwidth
        reader.f32()  # desired upper bandwidth
        reader.i16()  # notch filter mode
        reader.f32()  # actual impedance test frequency
        reader.f32()  # desired impedance test frequency
        if is_rhs:
            reader.i16()  # amplifier settle mode
            reader.i16()  # charge recovery mode
            reader.f32()  # stim step size
            reader.f32()  # charge recovery current limit
            reader.f32()  # charge recovery target voltage
        for _ in range(3):
            reader.qstring()  # notes

        dc_saved = bool(reader.i16()) if is_rhs else False
        temp_count = reader.i16() if (not is_rhs and version > 1.09) else 0
        if not 0 <= temp_count <= 64:
            raise ValueError(f"{path}: invalid Intan temperature sensor count {temp_count}")
        if is_rhs or version > 1.29:
            reader.i16()  # board mode
        if is_rhs or version > 1.99:
            reader.qstring()  # reference channel name

        num_groups = reader.i16()
        if not 0 <= num_groups <= 12:
            raise ValueError(f"{path}: invalid Intan signal group count {num_groups}")

        labels = []
        counts = {}
        for _group in range(num_groups):
            reader.qstring()  # group name
            reader.qstring()  # group prefix
            reader.i16()  # group enabled
            num_channels = reader.i16()
            if not 0 <= num_channels <= 520:
                raise ValueError(
                    f"{path}: invalid Intan signal group channel count {num_channels}"
                )
            reader.i16()  # group amplifier channel count
            for _channel in range(num_channels):
                native_name = reader.qstring()
                custom_name = reader.qstring()
                reader.i16()  # native order
                reader.i16()  # custom order
                signal_type = reader.i16()
                enabled = reader.i16()
                reader.i16()  # chip channel
                if is_rhs:
                    reader.i16()  # command stream
                reader.i16()  # board stream
                for _ in range(4):
                    reader.i16()  # spike scope trigger settings
                reader.f32()  # impedance magnitude
                reader.f32()  # impedance phase
                if not enabled:
                    continue
                if is_rhs:
                    valid_type = signal_type in range(7) or signal_type in (10, 11)
                else:
                    valid_type = signal_type in range(6)
                if not valid_type:
                    raise ValueError(
                        f"{path}: unknown Intan signal type {signal_type}"
                    )
                counts[signal_type] = counts.get(signal_type, 0) + 1
                if signal_type == 0:
                    labels.append(custom_name or native_name or f"CH{len(labels):03d}")

        header_size = reader.tell()

    num_amp = counts.get(0, 0)
    if num_amp <= 0:
        raise ValueError(f"{path}: no enabled amplifier channels")
    samples = 60 if (not is_rhs and main_version == 1) else 128
    dtype = _block_dtype(samples, num_amp, is_rhs, dc_saved, counts, temp_count)
    stride = dtype.itemsize

    data_bytes = path.stat().st_size - header_size
    if data_bytes < 0 or data_bytes % stride:
        raise ValueError(
            f"{path}: {data_bytes} data bytes do not fit Intan block size {stride} "
            "(truncated or corrupt file)"
        )
    num_blocks = data_bytes // stride

    return IntanFile(
        path=path,
        sample_rate=float(sample_rate),
        labels=labels,
        num_channels=num_amp,
        header_size=header_size,
        samples_per_block=samples,
        num_blocks=num_blocks,
        stride=stride,
        dtype=dtype,
    )


