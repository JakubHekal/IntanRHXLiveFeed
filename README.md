# LEECH

Live Electrophysiology Equipment Capture Hub

Real-time data acquisition, visualization, and recording for electrophysiology hardware.

## Supported Devices

- **Intan RHX** -- up to 128 channels via TCP/IP
- **miniSMU MS01** -- source-measure unit via USB or TCP

## Features

- Real-time signal streaming and visualization (raw waveforms, PSD, spike counts, averaged waveforms)
- Experiment sequencing with drag-and-drop timeline editor
- Marker management (add, rename, delete, embedded in CSV recordings)
- Run history with replay support
- Dark theme

## Quick Start

### Requirements

- Python 3.8+
- For Intan RHX: Intan software running with TCP server enabled

### Install

```bash
pip install -r requirements.txt
```

### Run

```bash
python main.py
```

## Build Executable

```bash
python pyinstall_run.py
```

The executable will be placed in the `dist/` directory.

## Pre-built Executable

Pre-built executables are available on the [Releases page](https://github.com/JakubHekal/IntanRHXLiveFeed/releases). Download the latest version and run it directly -- no Python installation required.
