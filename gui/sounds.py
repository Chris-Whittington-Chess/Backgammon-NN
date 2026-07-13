"""Tiny synthesized sound effects for the GUI.

Generates two short WAV files on first use (no external assets needed) and plays
them via QtMultimedia. All audio is best-effort: if there is no audio device or
QtMultimedia backend, playback is silently skipped.
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path

ASSETS = Path(__file__).resolve().parent / "assets"
RATE = 44100


def _write_wav(path: Path, samples):
    path.parent.mkdir(exist_ok=True)
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        frames = b"".join(struct.pack("<h", max(-32767, min(32767, int(s)))) for s in samples)
        w.writeframes(frames)


def _noise_burst(n, decay, seed):
    """A short decaying noise burst (a 'clack')."""
    import random

    rng = random.Random(seed)
    return [rng.uniform(-1, 1) * (2.718 ** (-decay * i / n)) for i in range(n)]


def ensure_assets():
    dice = ASSETS / "dice.wav"
    place = ASSETS / "place.wav"
    if not place.exists():
        # A soft tock: quick noise transient over a low decaying tone.
        import math

        n = int(RATE * 0.09)
        tone = [math.sin(2 * math.pi * 180 * i / RATE) * (2.718 ** (-30 * i / n)) for i in range(n)]
        noise = _noise_burst(n, 55, 1)
        _write_wav(place, [22000 * (0.5 * t + 0.5 * z) for t, z in zip(tone, noise)])
    if not dice.exists():
        # Three clacks with small gaps — dice tumbling and landing.
        gap = [0.0] * int(RATE * 0.06)
        clack1 = [18000 * s for s in _noise_burst(int(RATE * 0.05), 40, 2)]
        clack2 = [20000 * s for s in _noise_burst(int(RATE * 0.05), 38, 3)]
        clack3 = [24000 * s for s in _noise_burst(int(RATE * 0.07), 30, 4)]
        _write_wav(dice, clack1 + gap + clack2 + gap + clack3)
    return dice, place


class Sfx:
    """Loads and plays the effects; degrades gracefully without audio."""

    def __init__(self):
        self.dice = self.place = None
        try:
            from PySide6.QtCore import QUrl
            from PySide6.QtMultimedia import QSoundEffect

            dice_path, place_path = ensure_assets()
            self.dice = QSoundEffect()
            self.dice.setSource(QUrl.fromLocalFile(str(dice_path)))
            self.place = QSoundEffect()
            self.place.setSource(QUrl.fromLocalFile(str(place_path)))
        except Exception:
            self.dice = self.place = None

    def _play(self, eff):
        try:
            if eff is not None:
                eff.play()
        except Exception:
            pass

    def play_dice(self):
        self._play(self.dice)

    def play_place(self):
        self._play(self.place)
