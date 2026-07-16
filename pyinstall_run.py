import PyInstaller.__main__
import os
from PyInstaller.utils.hooks import collect_submodules

project_root = os.path.dirname(os.path.abspath(__file__))

# Dynamically gather your internal modules so you don't miss any
try:
    rhx_hidden_imports = collect_submodules('leech')
except Exception:
    rhx_hidden_imports = [
        'leech.device._rhx_config', 'leech.device._rhx_device',
        'leech.state_machine', 'leech.state_manager', 'leech.telemetry_logger',
        'leech.workers.rhx_worker', 'leech.workers.processing_worker',
        'leech.workers.chunk_writer', 'leech.workers.marker_manager',
        'leech.processing.psd', 'leech.processing.spike_count',
        'leech.processing.spike_plot', 'leech.screens.connect_screen',
        'leech.screens.plot_screen', 'leech.screens.marker_dialog',
    ]

# Aggressive Exclusions to strip out bloat
exclusions = [
    # 1. AI / ML / Heavy Math (If you don't use them directly)
    'torch', 'tensorflow', 'sklearn', 'numba', 'llvmlite', 'h5py', 'seaborn',

    # 2. Unused UI / Qt Frameworks (You use PyQt5, so block the others)
    'PyQt6', 'PySide2', 'PySide6', 'tkinter', 'wx', 
    'PyQt5.QtWebEngine', 'PyQt5.QtWebEngineCore', 'PyQt5.QtWebEngineWidgets', 
    'PyQt5.QtWebChannel', 'PyQt5.QtWebSockets', 'PyQt5.QtNetwork', 'PyQt5.QtSql',
    'pyqtgraph.examples',

    # 3. Jupyter, IPython, and debugging bloat (Pulled in by pandas/matplotlib)
    'IPython', 'jupyter_client', 'jupyter_core', 'traitlets', 'prompt_toolkit', 
    'zmq', 'jedi', 'pygments', 'debugpy', 'pdb',

    # 4. Database & Cloud stuff (Pulled in by pandas)
    'sqlalchemy', 'sqlite3', 'boto3', 'botocore', 's3fs', 'fsspec', 'pyarrow', 
    'fastparquet', 'tables',

    # 5. Unused file format parsers (Pulled in by pandas / scipy)
    'openpyxl', 'xlrd', 'pyxlsb', 'odf', 'jinja2', 'bs4', 'lxml', 'beautifulsoup4', 
    'html5lib', 'markupsafe', 'yaml', 'toml',

    # 6. Web Servers & Networking (Pulled in by matplotlib web backends)
    'tornado', 'aiohttp', 'urllib3', 'requests'
]

pyinstaller_args = [
    os.path.join(project_root, 'main.py'),
    '--name', 'Leech',
    '--icon', os.path.join(project_root, 'assets', 'icon.ico'),
    '--splash', os.path.join(project_root, 'assets', 'icon.png'),
    '--onefile',
    '--windowed',
    '--noconfirm',
    '--clean',
    '--upx-dir', '', 
    '--paths', project_root,
]

# Append exclusions
for mod in exclusions:
    pyinstaller_args.extend(['--exclude-module', mod])

# Append hidden imports
for mod in rhx_hidden_imports:
    pyinstaller_args.extend(['--hidden-import', mod])

print("Building Leech with PyInstaller...")
PyInstaller.__main__.run(pyinstaller_args)