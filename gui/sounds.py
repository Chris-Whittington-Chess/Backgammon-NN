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


def _clack(n, seed, base=760.0, bright=0.30):
    """One die landing: an ivory-ish clack rather than a click.

    Broadband noise on its own reads as a synthetic tick. Real dice are small
    dense objects that *ring* briefly, so the body here is three damped
    sinusoidal modes (an inharmonic stack, as a struck solid gives) and the noise
    is only a brief contact transient, low-passed to take the fizz off the top.
    """
    rng = random.Random(seed)
    # (frequency, amplitude, decay) — higher modes are quieter and die faster.
    modes = ((base, 1.0, 26.0), (base * 1.62, 0.5, 34.0), (base * 2.31, 0.22, 46.0))
    out = []
    lp = 0.0
    for i in range(n):
        t = i / RATE
        s = sum(a * math.sin(2 * math.pi * f * t) * math.exp(-d * t) for f, a, d in modes)
        # One-pole low-pass on the noise: keeps the contact, drops the hiss.
        lp += 0.22 * (rng.uniform(-1, 1) - lp)
        s += bright * lp * math.exp(-90.0 * t)
        out.append(s)
    peak = max(abs(v) for v in out) or 1.0
    return [v / peak for v in out]


def ensure_assets():
    # The filename carries the generation: bumping it retires any older cached
    # roll (v3 = warmer, ivory clacks instead of noise bursts).
    dice = ASSETS / "dice3.wav"
    place = ASSETS / "place.wav"
    for stale in ASSETS.glob("dice*.wav"):
        if stale != dice:
            stale.unlink()
    if not place.exists():
        # A soft tock: quick noise transient over a low decaying tone.
        import math

        n = int(RATE * 0.09)
        tone = [math.sin(2 * math.pi * 180 * i / RATE) * (2.718 ** (-30 * i / n)) for i in range(n)]
        noise = _noise_burst(n, 55, 1)
        _write_wav(place, [22000 * (0.5 * t + 0.5 * z) for t, z in zip(tone, noise)])
    if not dice.exists():
        # A longer tumble: a rapid rattle, then several clacks with growing gaps
        # as the dice settle — about 1.1 s total.
        samples = []
        # Rattle: quick, light taps, pitched high — dice knocking in the hand.
        # Detuning each keeps it from sounding like one sample repeated.
        for k in range(9):
            samples += [9000 * s for s in
                        _clack(int(RATE * 0.026), 10 + k, base=1180 + 90 * (k % 4), bright=0.22)]
            samples += [0.0] * int(RATE * 0.028)
        # Landing: fuller and lower, spacing out as they come to rest.
        for k, (dur, amp, gap, base) in enumerate(
            [(0.10, 17000, 0.07, 880), (0.11, 20000, 0.09, 760),
             (0.13, 23000, 0.12, 690), (0.16, 21000, 0.0, 620)]
        ):
            samples += [amp * s for s in _clack(int(RATE * dur), 30 + k, base=base)]
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
