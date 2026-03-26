import PyInstaller.__main__
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python-intan'))

PyInstaller.__main__.run([
    'main.py',
    '--name', 'rhx_realtime_feed',
    '--onefile',
    '--windowed',
    '--exclude-module', 'torch',
    '--exclude-module', 'tensorflow',
    '--exclude-module', 'sklearn',
    '--exclude-module', 'pysqlite3',
    '--exclude-module', 'seaborn',
    '--paths', './python-intan',
    '--hidden-import', 'intan',
    '--hidden-import', 'intan.interface'
])