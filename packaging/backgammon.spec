# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the standalone Backgammon app.

Produces a single self-contained `Backgammon.exe`: the PySide6 GUI, the Rust
engine (`bgcore.pyd`, which embeds the ONNX runtime), and the trained net
(`td.onnx`). No Python install, no PyTorch, nothing to unpack.

The app plays via `NativeNeuralEngine`/`RolloutEngine`, which run the net inside
`bgcore.pyd` — so torch, numpy and onnxruntime are all excluded below. The
trainer-side imports of those live inside functions the frozen app never calls
(`app.py` picks the native engine when `bgcore.Neural` exists), so dropping them
is safe; keeping them would add hundreds of MB.

Build via `packaging/build.py`, not PyInstaller directly — it generates the two
inputs referenced below that the repo doesn't track (the synthesized sound assets
and the icon), and it verifies the packaged exe afterwards.

Build:  .venv/Scripts/python packaging/build.py
Output: dist/Backgammon.exe
"""

from pathlib import Path

ROOT = Path(SPECPATH).parent  # noqa: F821 — SPECPATH is injected by PyInstaller

a = Analysis(
    [str(ROOT / "gui" / "app.py")],
    pathex=[str(ROOT / "gui"), str(ROOT / "trainer")],
    binaries=[],
    datas=[
        # Matches the layout `_root()` expects under sys._MEIPASS.
        (str(ROOT / "models" / "td.onnx"), "models"),        # champion / contact
        (str(ROOT / "models" / "td_race.onnx"), "models"),   # phase engine: race net
        (str(ROOT / "gui" / "assets"), "assets"),
    ],
    # The engine extension, plus the trainer-side modules app.py imports by name.
    hiddenimports=["bgcore", "cube", "engine_api", "sounds"],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # Training-only — the frozen app never constructs the torch engine.
        "torch",
        "numpy",
        "onnxruntime",
        "onnx",
        "matplotlib",
        "scipy",
        "PIL",
        # Not used by this GUI.
        "tkinter",
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineWidgets",
        "PySide6.QtQuick",
        "PySide6.QtQml",
        "PySide6.Qt3DCore",
        "PySide6.QtCharts",
        "PySide6.QtDataVisualization",
        "PySide6.QtPdf",
        "PySide6.QtDesigner",
        "PySide6.QtTest",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="Backgammon",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=False,  # windowed app — no console flash on launch
    disable_windowed_traceback=False,
    icon=str(ROOT / "packaging" / "backgammon.ico"),
)
