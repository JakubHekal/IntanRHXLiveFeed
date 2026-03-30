import PyInstaller.__main__
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python-intan'))

pyinstaller_args = [
    'main.py',
    '--name', 'rhx_realtime_feed',
    '--onefile',
    '--windowed',
    '--noconfirm',
    '--clean',
    '--optimize', '2',
    '--exclude-module', 'torch',
    '--exclude-module', 'tensorflow',
    '--exclude-module', 'sklearn',
    '--exclude-module', 'numba',
    '--exclude-module', 'llvmlite',
    '--exclude-module', 'h5py',
    '--exclude-module', 'pysqlite3',
    '--exclude-module', 'seaborn',
    '--exclude-module', 'pyqtgraph.examples',
    '--exclude-module', 'PyQt5.QtWebEngine',
    '--exclude-module', 'PyQt5.QtWebEngineCore',
    '--exclude-module', 'PyQt5.QtWebEngineWidgets',
    '--exclude-module', 'PyQt5.QtWebChannel',
    '--exclude-module', 'PyQt5.QtWebSockets',
    '--paths', './python-intan',
    '--hidden-import', 'intan',
    '--hidden-import', 'intan.io',
    '--hidden-import', 'intan.interface'
]

PyInstaller.__main__.run(pyinstaller_args)