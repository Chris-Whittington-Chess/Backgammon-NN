"""Tiny synthesized sound effects for the GUI.

Generates two short WAV files on first use (no external assets needed) and plays
them by pushing their PCM straight at the audio device via QAudioSink.

We deliberately do *not* use QSoundEffect. It decodes through Qt's media backend,
and on some Windows setups that path reports Status.Ready and isPlaying() == True
while producing no sound at all — silent, and it lies about it. We synthesise the
samples ourselves, so there is nothing to decode: handing raw PCM to QAudioSink
skips the whole decoder.

All audio is best-effort: with no device or no QtMultimedia, playback is skipped.
"""

from __future__ import annotations

import math
import random
import struct
import sys
import wave
from pathlib import Path

# Frozen (PyInstaller) builds ship the WAVs and unpack them under `sys._MEIPASS`;
# from source they live next to this file and are generated on first use.
_BASE = getattr(sys, "_MEIPASS", None)
ASSETS = (Path(_BASE) if _BASE else Path(__file__).resolve().parent) / "assets"
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
    rng = random.Random(seed)
    return [rng.uniform(-1, 1) * (2.718 ** (-decay * i / n)) for i in range(n)]


def _bandpass(sig, f0, q):
    """Two-pole band-pass (RBJ biquad). Shapes noise into a woody knock."""
    w0 = 2 * math.pi * f0 / RATE
    alpha = math.sin(w0) / (2 * q)
    b0, b1, b2 = alpha, 0.0, -alpha
    a0, a1, a2 = 1 + alpha, -2 * math.cos(w0), 1 - alpha
    b0, b1, b2, a1, a2 = b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0
    out = []
    x1 = x2 = y1 = y2 = 0.0
    for x in sig:
        y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        out.append(y)
        x2, x1 = x1, x
        y2, y1 = y1, y
    return out


def _clack(n, seed, f0=1300.0, q=1.5, decay=60.0):
    """One die knock: noise shaped by a broad resonance.

    Dice have no pitch — they knock. Building a clack from a stack of harmonic
    modes gives it a definite note, and several such clacks in a row turn into a
    tune, which is exactly what you don't want. So this is noise pushed through a
    *wide* band-pass: enough colour to read as a small hard object on wood, too
    broad to carry a pitch, and not the raw white hiss that reads as a click.
    """
    rng = random.Random(seed)
    noise = [rng.uniform(-1, 1) for _ in range(n)]
    band = _bandpass(noise, f0, q)
    out = [v * math.exp(-decay * i / RATE) for i, v in enumerate(band)]
    peak = max(abs(v) for v in out) or 1.0
    return [v / peak for v in out]


def ensure_assets():
    # The filename carries the generation: bumping it retires any older cached
    # roll (v4 = band-passed knocks; v3's harmonic modes rang out as a tune).
    dice = ASSETS / "dice4.wav"
    place = ASSETS / "place2.wav"
    for stale in list(ASSETS.glob("dice*.wav")) + list(ASSETS.glob("place*.wav")):
        if stale not in (dice, place):
            stale.unlink()
    if not place.exists():
        # A checker landing: the same woody knock as the dice, lower and shorter.
        # The old one was a 90ms tone-and-hiss at half amplitude, which was all
        # but inaudible at a sane volume. The silent tail matters: the sink is
        # created per play, and a buffer this short can otherwise run out before
        # the device has finished opening.
        samples = [27000 * s for s in _clack(int(RATE * 0.075), 1, f0=950, q=1.4, decay=75)]
        samples += [0.0] * int(RATE * 0.06)
        _write_wav(place, samples)
    if not dice.exists():
        # A tumble: a rapid rattle, then a few knocks with growing gaps as the
        # dice settle — about 1.1 s total. Every knock's centre frequency is
        # jittered rather than stepped: a tidy sequence of centres would be heard
        # as a melody, which no dice roll has.
        rng = random.Random(4)
        samples = []
        # Rattle: light, quick, high knocks — dice shaken together.
        for k in range(10):
            samples += [7000 * s for s in _clack(
                int(RATE * 0.020), 10 + k, f0=rng.uniform(1500, 2600), q=1.3, decay=110)]
            samples += [0.0] * int(RATE * 0.026)
        # Landing: heavier, lower knocks as they hit the board and come to rest.
        for k, (dur, amp, gap) in enumerate(
            [(0.05, 17000, 0.075), (0.055, 20000, 0.10), (0.06, 23000, 0.13), (0.07, 21000, 0.0)]
        ):
            samples += [amp * s for s in _clack(
                int(RATE * dur), 30 + k, f0=rng.uniform(800, 1500), q=1.5, decay=65)]
            samples += [0.0] * int(RATE * gap)
        _write_wav(dice, samples)
    return dice, place


def _pcm_from_wav(path):
    """The raw 16-bit mono samples of a WAV, ready to hand to the device."""
    with wave.open(str(path)) as w:
        if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (1, 2, RATE):
            raise ValueError(f"{path.name}: expected mono 16-bit {RATE}Hz")
        return w.readframes(w.getnframes())


class _Voice:
    """One effect: its PCM, and the sink currently playing it.

    A fresh QAudioSink per play keeps overlapping effects independent. Qt does
    not take ownership of the byte array or buffer, so this holds them for as
    long as the sink might read from them — drop the references and playback goes
    silent or crashes.
    """

    def __init__(self, pcm, fmt, device):
        self._pcm, self._fmt, self._device = pcm, fmt, device
        self._sink = self._buf = self._ba = None

    def play(self, volume):
        from PySide6.QtCore import QBuffer, QByteArray
        from PySide6.QtMultimedia import QAudioSink

        self.stop()
        self._ba = QByteArray(self._pcm)
        self._buf = QBuffer(self._ba)
        self._buf.open(QBuffer.ReadOnly)
        self._sink = QAudioSink(self._device, self._fmt)
        self._sink.setVolume(volume)
        self._sink.start(self._buf)

    def stop(self):
        try:
            if self._sink is not None:
                self._sink.stop()
        except Exception:
            pass
        self._sink = self._buf = self._ba = None

    def is_playing(self) -> bool:
        # Qt 6.7 renamed the QAudio namespace to QtAudio, so the enum you import
        # may not be the one state() returns and `==` quietly fails. Compare by
        # name, which holds either way.
        return self._sink is not None and str(self._sink.state()).endswith("ActiveState")


class Sfx:
    """Loads and plays the effects; degrades gracefully without audio."""

    def __init__(self, volume: float = 0.5):
        self.dice = self.place = None
        self.device_name = ""
        self._volume = max(0.0, min(1.0, float(volume)))
        try:
            from PySide6.QtMultimedia import QAudioFormat, QMediaDevices

            fmt = QAudioFormat()
            fmt.setSampleRate(RATE)
            fmt.setChannelCount(1)
            fmt.setSampleFormat(QAudioFormat.Int16)
            device = QMediaDevices.defaultAudioOutput()
            if device is None or device.isNull() or not device.isFormatSupported(fmt):
                return
            self.device_name = device.description()
            dice_path, place_path = ensure_assets()
            self.dice = _Voice(_pcm_from_wav(dice_path), fmt, device)
            self.place = _Voice(_pcm_from_wav(place_path), fmt, device)
        except Exception:
            self.dice = self.place = None

    @property
    def ok(self) -> bool:
        return self.dice is not None

    @property
    def volume(self) -> float:
        return self._volume

    def set_volume(self, v: float) -> None:
        """Set playback volume, 0.0 (muted) to 1.0. Takes effect on the next
        sound; the sinks are created per play."""
        self._volume = max(0.0, min(1.0, float(v)))

    def _play(self, voice):
        try:
            if voice is not None and self._volume > 0.0:
                voice.play(self._volume)
        except Exception:
            pass

    def play_dice(self):
        self._play(self.dice)

    def play_place(self):
        self._play(self.place)

    def is_playing(self) -> bool:
        return any(v is not None and v.is_playing() for v in (self.dice, self.place))
