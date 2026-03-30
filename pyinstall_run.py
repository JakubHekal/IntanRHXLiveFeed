import PyInstaller.__main__
import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
intan_root = os.path.join(project_root, 'python-intan')
intan_pkg_root = os.path.join(intan_root, 'intan')

if not os.path.isdir(intan_pkg_root):
    raise FileNotFoundError(f"Missing local intan package directory: {intan_pkg_root}")

sys.path.insert(0, intan_root)

data_sep = ';' if sys.platform.startswith('win') else ':'

pyinstaller_args = [
    os.path.join(project_root, 'main.py'),
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
    '--paths', intan_root,
    '--add-data', f'{intan_pkg_root}{data_sep}python-intan/intan',
    '--hidden-import', 'intan',
    '--hidden-import', 'intan.io',
    '--hidden-import', 'intan.interface'
]

PyInstaller.__main__.run(pyinstaller_args)