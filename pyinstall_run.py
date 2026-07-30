import PyInstaller.__main__
import os
from PyInstaller.utils.hooks import collect_submodules

project_root = os.path.dirname(os.path.abspath(__file__))

# ponytail: dynamic collection primary, hardcoded fallback
try:
    hidden_imports = collect_submodules('leech')
    if not hidden_imports:
        raise ValueError("empty")
except Exception:
    hidden_imports = [
        'leech', 'leech.ui',
        'leech.telemetry_logger', 'leech.plot_settings', 'leech.updater',
        'leech.device.base', 'leech.device.ring_buffer',
        'leech.device.background_worker', 'leech.device.widget_builder',
        'leech.device.tabs.base',
        'leech.device.intan_rhx.device', 'leech.device.intan_rhx.tab',
        'leech.device.intan_rhx.canvas', 'leech.device.intan_rhx._processing_tasks',
        'leech.device.intan_rhx.processing.psd',
        'leech.device.intan_rhx.processing.spike_count',
        'leech.device.intan_rhx.processing.spike_plot',
        'leech.device.minismu.device', 'leech.device.minismu.tab',
        'leech.device.minismu.canvas',
        'leech.device.simulated.device', 'leech.device.simulated.tab',
        'leech.screens.stage', 'leech.screens.timeline',
        'leech.screens.plot_screen', 'leech.screens.plot_helpers',
        'leech.screens.marker_dialog', 'leech.screens.channel_selector',
        'leech.screens._registry',
        'leech.workers.chunk_writer', 'leech.workers.replay_worker',
        'leech.experiment.experiment', 'leech.experiment.experiment_dialog',
        'leech.experiment.experiment_runner',
    ]

# Exclusions to strip bloat from the bundle
exclusions = [
    # AI / ML / Heavy Math
    'torch', 'tensorflow', 'sklearn', 'numba', 'llvmlite', 'h5py', 'seaborn',

    # Unused UI / Qt frameworks (we use PyQt5)
    'PyQt6', 'PySide2', 'PySide6', 'tkinter', 'wx',
    'PyQt5.QtWebEngine', 'PyQt5.QtWebEngineCore', 'PyQt5.QtWebEngineWidgets',
    'PyQt5.QtWebChannel', 'PyQt5.QtWebSockets', 'PyQt5.QtNetwork', 'PyQt5.QtSql',
    'pyqtgraph.examples',

    # matplotlib — only used in standalone __main__ scripts, not at runtime
    'matplotlib', 'mpl_toolkits',

    # Jupyter / IPython / debugging (pulled in by pandas)
    'IPython', 'jupyter_client', 'jupyter_core', 'traitlets', 'prompt_toolkit',
    'zmq', 'jedi', 'pygments', 'debugpy', 'pdb',

    # Database & Cloud (pulled in by pandas)
    'sqlalchemy', 'sqlite3', 'boto3', 'botocore', 's3fs', 'fsspec', 'pyarrow',
    'fastparquet', 'tables',

    # Unused file format parsers (pulled in by pandas / scipy)
    'openpyxl', 'xlrd', 'pyxlsb', 'odf', 'jinja2', 'bs4', 'lxml', 'beautifulsoup4',
    'html5lib', 'markupsafe', 'yaml', 'toml',

    # Web servers & networking (pulled in by matplotlib web backends)
    'tornado', 'aiohttp', 'urllib3', 'requests',

    # Packaging / setup (not needed at runtime)
    'setuptools', 'pkg_resources',
]

pyinstaller_args = [
    os.path.join(project_root, 'main.py'),
    '--name', 'LEECH',
    '--icon', os.path.join(project_root, 'assets', 'icon.ico'),
    '--splash', os.path.join(project_root, 'assets', 'icon.png'),
    '--add-data', f'{os.path.join(project_root, "assets", "icon.png")};.',
    '--onefile',
    '--windowed',
    '--noconfirm',
    '--clean',
    '--paths', project_root,
]

for mod in exclusions:
    pyinstaller_args.extend(['--exclude-module', mod])

for mod in hidden_imports:
    pyinstaller_args.extend(['--hidden-import', mod])

print("Building LEECH with PyInstaller...")
PyInstaller.__main__.run(pyinstaller_args)