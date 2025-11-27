# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_data_files
from PyInstaller.utils.hooks import copy_metadata
datas = [
    ('src/alembic/*', 'src/alembic'),
    ('src/alembic.ini', '.'),
]

datas += copy_metadata("requests")
datas += copy_metadata("diffusers")
datas += copy_metadata("huggingface-hub")
datas += copy_metadata("filelock")
datas += copy_metadata("imageio")

binaries = []
hiddenimports = []
hiddenimports += ["aiosqlite","imageio"]

tmp_ret = collect_all('getiaction') + collect_all('torch') + collect_all('imageio')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

a = Analysis(
    ['src/main.py'],
    pathex=['src'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['pyinstaller/runtime-hook.py','pyinstaller/hook-setenv.py'],
    excludes=['torch.utils.benchmark'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='geti-action-backend',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='geti-action-backend',
)
