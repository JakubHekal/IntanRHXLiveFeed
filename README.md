# RHX Realtime Feed

Stream, visualize, and analyze neural recordings from Intan RHX systems in real time.

## Features

- Connects to Intan RHX systems via TCP
- Real-time streaming with configurable sample rates
- PSD (Power Spectral Density) visualization
- Spike detection with average waveform plotting
- Marker management (add, rename, delete)
- Snapshot capture (PSD and waveforms)
- Project-based recording with structured storage
- State-machine-driven lifecycle management
- Telemetry logging for session diagnostics

## Quick Start (from source)

### Requirements

- Python 3.8+
- Intan RHX software running with TCP server enabled

### Installation

```bash
pip install -r requirements.txt
```

### Usage

```bash
python main.py
```

## Pre-built Executable

Pre-built executables are available on the [Releases page](https://github.com/JakubHekal/IntanRHXLiveFeed/releases). Download the latest version and run it directly -- no Python installation required.

## Build Executable from Source

To build a standalone executable with PyInstaller:

```bash
python pyinstall_run.py
```

The executable will be placed in the `dist/` directory.
