import PyInstaller.__main__
import sys
import os
import importlib.util

project_root = os.path.dirname(os.path.abspath(__file__))
intan_root = os.path.join(project_root, 'python-intan')
intan_pkg_root = os.path.join(intan_root, 'intan')
use_local_intan = os.path.isdir(intan_pkg_root)

if use_local_intan:
    sys.path.insert(0, intan_root)
else:
    # Fallback for CI runs where python-intan submodule is not checked out.
    if importlib.util.find_spec('intan') is None:
        raise FileNotFoundError(
            "Local intan package not found and installed 'intan' package is unavailable. "
            "Either checkout submodules (git submodule update --init --recursive) "
            "or install python-intan in the build environment."
        )

data_sep = ';' if sys.platform.startswith('win') else ':'

pyinstaller_args = [
    os.path.join(project_root, 'main.py'),
    '--name', 'RHX Realtime Feed',
    '--icon', os.path.join(project_root, 'assets', 'icon.ico'),
    '--splash', os.path.join(project_root, 'assets', 'RHX_splash.png'),
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
    '--hidden-import', 'intan',
    '--hidden-import', 'intan.io',
    '--hidden-import', 'intan.interface'
]

if use_local_intan:
    pyinstaller_args.extend([
        '--paths', intan_root,
        '--add-data', f'{intan_pkg_root}{data_sep}python-intan/intan',
    ])

PyInstaller.__main__.run(pyinstaller_args)