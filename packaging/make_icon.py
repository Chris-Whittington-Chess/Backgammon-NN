"""Generate `packaging/backgammon.ico` — the app icon for the packaged build.

Draws a board motif (two points and a checker) in the GUI's own palette, then
assembles a multi-size PNG-in-ICO so Windows has a crisp image at every size it
asks for (16px in the taskbar up to 256px in Explorer).

Run: .venv/Scripts/python packaging/make_icon.py
"""

from __future__ import annotations

import struct
from pathlib import Path

from PySide6.QtCore import QBuffer, QByteArray, QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPolygonF

OUT = Path(__file__).resolve().parent / "backgammon.ico"

# The GUI's palette (gui/app.py), so the icon matches the board it opens.
FRAME = QColor("#4a2f1a")
FELT = QColor("#0f5132")
PT_LIGHT = QColor("#d9b382")
ENGINE = QColor("#9e2b25")
HUMAN = QColor("#f2ead6")
HUMAN_EDGE = QColor("#b9a97e")

S = 256  # master size; every icon size is a smooth downscale of this
SIZES = [256, 128, 64, 48, 32, 16]


def render(size: int = S) -> QImage:
    img = QImage(size, size, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    k = size / S  # scale factor from the 256-unit design grid

    # Wooden frame, then the felt playing surface inset within it.
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(FRAME))
    p.drawRoundedRect(QRectF(0, 0, size, size), 40 * k, 40 * k)
    p.setBrush(QBrush(FELT))
    p.drawRoundedRect(QRectF(18 * k, 18 * k, size - 36 * k, size - 36 * k), 20 * k, 20 * k)

    def tri(pts, color):
        p.setBrush(QBrush(color))
        p.drawPolygon(QPolygonF([QPointF(x * k, y * k) for x, y in pts]))

    # A light point hanging from the top, a red point rising from the bottom.
    tri([(24, 24), (124, 24), (74, 172)], PT_LIGHT)
    tri([(132, 232), (232, 232), (182, 84)], ENGINE)

    # An ivory checker over the middle — the piece that reads at 16px.
    p.setBrush(QBrush(HUMAN))
    p.setPen(QPen(HUMAN_EDGE, 7 * k))
    p.drawEllipse(QPointF(128 * k, 150 * k), 46 * k, 46 * k)
    p.end()
    return img


def png_bytes(img: QImage) -> bytes:
    # `ba` must outlive the buffer — QBuffer does not own it, and letting the
    # QByteArray be a temporary segfaults.
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QBuffer.WriteOnly)
    img.save(buf, "PNG")
    buf.close()
    return bytes(ba)


def main() -> None:
    master = render()
    images = [
        png_bytes(
            master.scaled(s, s, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            if s != S
            else master
        )
        for s in SIZES
    ]

    # ICONDIR: reserved=0, type=1 (icon), image count.
    out = bytearray(struct.pack("<HHH", 0, 1, len(SIZES)))
    offset = 6 + 16 * len(SIZES)
    for s, data in zip(SIZES, images):
        # ICONDIRENTRY; 0 means 256. PNG payloads are legal in ICO on Vista+.
        out += struct.pack(
            "<BBBBHHII", s % 256, s % 256, 0, 0, 1, 32, len(data), offset
        )
        offset += len(data)
    for data in images:
        out += data

    OUT.write_bytes(out)
    print(f"wrote {OUT} ({len(out):,} bytes; sizes {SIZES})")


if __name__ == "__main__":
    main()
