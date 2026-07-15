"""Build the standalone Backgammon app, then prove the result works.

This is the release build command — prefer it over invoking PyInstaller directly,
since it prepares two inputs the spec needs but the repo doesn't track (the
generated sound assets and the icon), and it verifies the packaged exe rather
than trusting that a successful build means a working app.

Steps: check prerequisites -> generate assets + icon -> PyInstaller -> selftest
the produced exe (loads the net, plays a move) -> report the artifact.

Run: .venv/Scripts/python packaging/build.py
Out: dist/Backgammon.exe
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXE = ROOT / "dist" / "Backgammon.exe"


def step(msg: str) -> None:
    print(f"\n==> {msg}", flush=True)


def fail(msg: str) -> "None":
    print(f"\nBUILD FAILED: {msg}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    os.chdir(ROOT)

    step("Checking prerequisites")
    onnx = ROOT / "models" / "td.onnx"
    if not onnx.exists():
        fail(f"{onnx} missing — export it first:\n"
             f"  .venv/Scripts/python trainer/export_onnx.py models/td_latest.pt")
    try:
        import bgcore
    except ImportError:
        fail("the bgcore extension isn't installed in this interpreter — build it:\n"
             "  cd crates/bgpy && ../../.venv/Scripts/maturin develop --release --features onnx")
    if not hasattr(bgcore, "Neural"):
        fail("bgcore lacks `Neural` — it was built without the onnx feature. Rebuild:\n"
             "  cd crates/bgpy && ../../.venv/Scripts/maturin develop --release --features onnx")
    print(f"  net   {onnx.relative_to(ROOT)} ({onnx.stat().st_size:,} bytes)")
    print(f"  engine {Path(bgcore.__file__).parent}")

    step("Generating sound assets (untracked — synthesized on demand)")
    sys.path.insert(0, str(ROOT / "gui"))
    from sounds import ensure_assets

    for p in ensure_assets():
        print(f"  {Path(p).relative_to(ROOT)}")

    step("Generating icon")
    subprocess.run([sys.executable, str(HERE / "make_icon.py")], check=True)

    step("Running PyInstaller (a few minutes)")
    proc = subprocess.run(
        [
            str(ROOT / ".venv" / "Scripts" / "pyinstaller.exe"),
            "--clean", "--noconfirm",
            "--distpath", str(ROOT / "dist"),
            "--workpath", str(ROOT / "build" / "pyinstaller"),
            str(HERE / "backgammon.spec"),
        ],
        capture_output=True,  # hundreds of INFO lines; surfaced only on failure
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stdout[-4000:])
        print(proc.stderr[-4000:], file=sys.stderr)
        fail(f"PyInstaller exited {proc.returncode}")
    if not EXE.exists():
        fail(f"{EXE} was not produced")

    step("Verifying the packaged exe (loads net, plays a move)")
    with tempfile.TemporaryDirectory() as td:
        report_path = Path(td) / "selftest.json"
        proc = subprocess.run([str(EXE), "--selftest", str(report_path)])
        if not report_path.exists():
            fail(f"selftest wrote no report (exit {proc.returncode}) — the exe likely "
                 f"crashed on startup")
        report = json.loads(report_path.read_text())

    if not report.get("ok"):
        fail("selftest failed:\n" + report.get("error", "(no detail)"))
    if not report.get("frozen"):
        fail("selftest ran unfrozen — wrong binary?")
    if report.get("torch_imported"):
        fail("torch got imported — the bundle is not torch-free")
    neural = [o for o in report["opponents"] if o.startswith("Neural")]
    if not neural:
        fail("no neural opponents — td.onnx did not load inside the bundle")
    if report.get("evaluator") != "NativeNeuralEngine":
        fail(f"expected the native engine, got {report.get('evaluator')}")
    # The app should open on its strongest opponent, which is the rollout engine
    # whenever the rollout bindings and the net are both present.
    if not report.get("default_opponent", "").startswith("Rollout"):
        fail(f"expected the app to default to the rollout engine, got "
             f"{report.get('default_opponent')!r}")
    # A build that ships silent audio looks fine from the outside: the old
    # QSoundEffect path reported Ready and isPlaying while emitting nothing. So
    # require the sink to actually go active on a real play.
    if not report.get("sound"):
        fail("no audio: the effects didn't load in the bundle")
    if not report.get("sound_plays"):
        fail(f"audio loaded but playing it did nothing — device "
             f"{report.get('sound_device')!r}, outputs {report.get('audio_outputs')}")

    for k in ("opponents", "default_opponent", "hint_engine", "evaluator",
              "engine", "best_move_31", "equity", "sound"):
        print(f"  {k:16} {report[k]}")

    size_mb = EXE.stat().st_size / 1e6
    print(f"\nBuild OK: {EXE.relative_to(ROOT)} ({size_mb:.1f} MB)")
    print("Verified: engine + net load inside the bundle, torch-free, window app.")


if __name__ == "__main__":
    main()
