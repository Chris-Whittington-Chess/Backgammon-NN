"""Desktop GUI for the backgammon engine (SPEC §10, milestone M6).

Play the trained neural net (0-ply / 1-ply), HCE, or Random. You are the ivory
side at the bottom, bearing off to the right.

Interaction:
  * Click the dice (or "Roll") to roll — the dice tumble briefly.
  * Left-click one of your checkers to select it (highlighted destinations
    appear), then left-click a destination (or the off tray) to move.
  * Right-click, or click the checker again, to deselect.
  The turn commits automatically once all playable dice are used, then the
  computer's reply is animated. The move panel logs every turn with its equity.

Run: .venv/Scripts/python gui/app.py
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

def _root() -> Path:
    """Base directory for bundled data (`models/`, `assets/`).

    When frozen by PyInstaller the app runs from a temp unpack dir exposed as
    `sys._MEIPASS`; from a source checkout it's the repo root.
    """
    bundle = getattr(sys, "_MEIPASS", None)
    return Path(bundle) if bundle else Path(__file__).resolve().parent.parent


ROOT = _root()
if not getattr(sys, "frozen", False):
    # In the bundle the trainer modules are packed as top-level modules already.
    sys.path.insert(0, str(ROOT / "trainer"))

from PySide6.QtCore import (
    Qt, QEventLoop, QObject, QPointF, QRectF, QRunnable, QSettings, QThreadPool,
    QTimer, Signal,
)
from PySide6.QtGui import (
    QAction, QBrush, QColor, QCursor, QFont, QPainter, QPen, QPolygonF,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

import bgcore
from cube import should_double, should_take
from engine_api import (
    HceEngine,
    NativeNeuralEngine,
    NativePhaseEngine,
    NeuralEngine,
    RandomEngine,
    RolloutEngine,
    format_steps,
)
from sounds import Sfx

BAR, OFF = bgcore.BAR, bgcore.OFF

FELT = QColor("#0f5132")
FRAME = QColor("#4a2f1a")
PT_LIGHT = QColor("#d9b382")
PT_DARK = QColor("#a9743f")
HUMAN = QColor("#f2ead6")
HUMAN_EDGE = QColor("#b9a97e")
ENGINE = QColor("#9e2b25")
ENGINE_EDGE = QColor("#511210")
SRC_RING = QColor("#ffd21f")
HUMAN_NUM = QColor("#e9dcc0")          # your point numbers
ENGINE_NUM = QColor("#b8635c")         # the engine's numbering of the same points
DEST_FILL = QColor(70, 220, 120, 150)


# What the help panel says. Kept next to the code it describes so it can't
# quietly drift out of date.
HELP_LINES = [
    # key=None spans the full width — an intro line, not a key/value row.
    (None, "Anything pulsing wants a click. Hover it and a box"),
    (None, "appears telling you what that click will do."),
    ("Start", "Click the counting dice: one die each, higher starts."),
    ("Roll", "Click the dice, or hover them for a Roll box."),
    ("Move", "Click one of your checkers, then a highlighted point."),
    ("", "Your movable checkers are ringed. Bear off to the tray."),
    ("Deselect", "Right-click, or click the same checker again."),
    ("Take back", "Undo (Ctrl+Z) — one checker at a time, back to"),
    ("", "the start of your turn. Your last die commits the turn."),
    ("Double", "Hover the cube for a Double box."),
    ("Answer", "When the cube pulses at you, hover it for Accept / Fold."),
    ("Hint", "Ranks your best moves with their equities."),
    ("Numbers", "Ivory = your point numbers, red = the engine's. Each side"),
    ("", "counts from its own home, so your 8 is its 17 (they total 25)."),
    ("", "Your moves read off ivory, the CPU's log lines off red."),
    ("Eval bar", "Right of the board: your live win chance. Ivory is you,"),
    ("", "rising from the bottom; red is the engine."),
    ("Pips", "Corner counts: how far each side has left to travel."),
    ("Cube", "Doubles the stakes. The number is the current multiplier."),
    ("Opponent", "Rollout is strongest; drop to 2/1/0-ply for an easier game."),
]


class _Task(QRunnable):
    """Runs one engine call off the UI thread.

    The engine can block for the best part of a second (rollouts), which froze
    the window solid. The Rust side releases the GIL for that work, so a worker
    thread genuinely runs in parallel rather than just moving the stall.
    """

    class _Signals(QObject):
        done = Signal(object, int)

    def __init__(self, fn, gen):
        super().__init__()
        self._fn, self._gen = fn, gen
        self.signals = self._Signals()

    def run(self):
        try:
            result = self._fn()
        except Exception as exc:      # deliver it; the UI thread decides what to do
            result = exc
        self.signals.done.emit(result, self._gen)


class HoverButton(QPushButton):
    """A button that reports hover, so the help panel can follow the cursor."""

    def __init__(self, text, on_hover):
        super().__init__(text)
        self._on_hover = on_hover

    def enterEvent(self, ev):
        self._on_hover(True)

    def leaveEvent(self, ev):
        self._on_hover(False)


class BoardView(QWidget):
    def __init__(self, on_click, on_dice, on_action=None):
        super().__init__()
        self.on_click, self.on_dice = on_click, on_dice
        self.on_action = on_action or (lambda key: None)
        self.setMinimumSize(760, 560)
        # Hover boxes appear without a button held down, so we need move events
        # delivered whenever the cursor is over the board.
        self.setMouseTracking(True)
        self.board = bgcore.Board.starting()
        self.dice: list[int] = []
        self.source_points: set[int] = set()
        self.dest_points: set[int] = set()
        self.carrying: int | None = None       # selected source point (ringed)
        self.floating: tuple[float, float, bool] | None = None  # engine-move sprite
        self.cube_value = 1
        self.cube_owner: int | None = None     # None centered, 0 you, 1 engine
        self.wink_dice = False                 # pulse the dice (your turn to roll)
        self.wink_cube = False                 # pulse the cube (engine has doubled)
        self.wink_on = True                    # current pulse phase
        self.opening = False                   # showing the opening-roll dice
        self.open_dice = None                  # (your_die, engine_die)
        self.open_resolved = False             # a winner has been thrown
        self.open_rolling = False              # clicked; throwing (maybe re-throwing)
        self.can_double = False                # you may offer a double right now
        self.hover_zone = None                 # "dice" | "cube" — which hover boxes show
        self.show_help = False                 # help panel overlays the board
        self.hint_rows = None                  # [(notation, equity)] while hinting
        self._geom = None

    # --- geometry ---
    def _layout(self):
        W, H = self.width(), self.height()
        margin = 16
        off_w = 44
        bar_w = 42
        pw = (W - 2 * margin - bar_w - off_w) / 12.0
        ph = (H - 2 * margin) * 0.44
        r = min(pw * 0.40, ph / 10.0)          # smaller: five fit without overlap
        left = margin
        xL = [left + c * pw for c in range(6)]
        bar_x0 = left + 6 * pw
        bar_x1 = bar_x0 + bar_w
        xL += [bar_x1 + c * pw for c in range(6)]
        off_x0 = xL[11] + pw
        s = min(46, pw)
        # The opening roll spreads the two dice apart (one each); keep the hit
        # area in step with what's drawn so the whole group stays clickable.
        spread = 2 * s + (34 if (self.opening and self.open_dice) else 12)
        dice_rect = QRectF(W * 0.70 - spread / 2, H / 2 - s / 2, spread, s)
        # The cube sits at the left edge, and moves to the owner's side.
        cr = r * 1.4
        cx = margin + pw * 0.5
        if self.cube_owner == 1:      # engine owns -> top
            cy = margin + ph + cr + 6
        elif self.cube_owner == 0:    # you own -> bottom
            cy = H - margin - ph - cr - 6
        else:                         # centered
            cy = H / 2
        return {
            "W": W, "H": H, "margin": margin, "pw": pw, "ph": ph, "r": r,
            "xL": xL, "bar": (bar_x0, bar_x1), "off": (off_x0, off_x0 + off_w),
            "dice_rect": dice_rect, "dice_s": s,
            "cube_rect": QRectF(cx - cr, cy - cr, 2 * cr, 2 * cr),
        }

    def _col_row(self, point):
        return (12 - point, False) if 1 <= point <= 12 else (point - 13, True)

    def _triangle(self, g, point):
        col, is_top = self._col_row(point)
        x0 = g["xL"][col]
        x1 = x0 + g["pw"]
        m, ph, H = g["margin"], g["ph"], g["H"]
        if is_top:
            return QPolygonF([QPointF(x0, m), QPointF(x1, m), QPointF((x0 + x1) / 2, m + ph)])
        return QPolygonF([QPointF(x0, H - m), QPointF(x1, H - m), QPointF((x0 + x1) / 2, H - m - ph)])

    def point_center(self, g, point, i, side=None):
        """Screen center of the i-th checker (0-based) on a display point, or the
        bar / off tray for point == BAR / OFF (side 0=human bottom, 1=engine top)."""
        if point == BAR:
            bx0, bx1 = g["bar"]
            cx = (bx0 + bx1) / 2
            cy = g["H"] * (0.62 if side == 0 else 0.30) + (i * 2 * g["r"] * (1 if side == 0 else -1))
            return cx, cy
        if point == OFF:
            ox0, ox1 = g["off"]
            return (ox0 + ox1) / 2, g["H"] * (0.80 if side == 0 else 0.20)
        col, is_top = self._col_row(point)
        cx = g["xL"][col] + g["pw"] / 2
        step = 2 * g["r"]
        cy = (g["margin"] + g["r"] + i * step) if is_top else (g["H"] - g["margin"] - g["r"] - i * step)
        return cx, cy

    # --- painting ---
    def paintEvent(self, _ev):
        g = self._geom = self._layout()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W, H, m = g["W"], g["H"], g["margin"]
        p.fillRect(self.rect(), FRAME)
        p.fillRect(QRectF(m, m, W - 2 * m, H - 2 * m), FELT)

        for point in range(1, 25):
            col, is_top = self._col_row(point)
            tri = self._triangle(g, point)
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(PT_LIGHT if (col + int(is_top)) % 2 == 0 else PT_DARK))
            p.drawPolygon(tri)
            if point in self.dest_points:
                p.setBrush(QBrush(DEST_FILL))
                p.drawPolygon(tri)

        bx0, bx1 = g["bar"]
        p.setBrush(QBrush(FRAME))
        p.setPen(Qt.NoPen)
        p.drawRect(QRectF(bx0, m, bx1 - bx0, H - 2 * m))

        self._draw_checkers(p, g)
        self._draw_bar(p, g)
        self._draw_off(p, g)
        self._draw_cube(p, g)
        self._draw_dice(p, g)
        self._draw_numbers(p, g)
        self._draw_pips(p, g)
        if self.floating:
            x, y, human = self.floating
            self._disc(p, x, y, g["r"], human)
        self._draw_hover(p, g)   # last — the boxes overlay the board
        self._draw_hint(p, g)
        self._draw_help(p, g)
        p.end()

    def _disc(self, p, cx, cy, r, human, label=None):
        p.setBrush(QBrush(HUMAN if human else ENGINE))
        p.setPen(QPen(HUMAN_EDGE if human else ENGINE_EDGE, 2))
        p.drawEllipse(QPointF(cx, cy), r, r)
        if label:
            p.setPen(QPen(QColor("#3a2f1a") if human else QColor("#ffffff")))
            p.setFont(QFont("Arial", int(r * 1.05), QFont.Bold))
            p.drawText(QRectF(cx - r, cy - r, 2 * r, 2 * r), Qt.AlignCenter, label)

    def _draw_checkers(self, p, g):
        for point in range(1, 25):
            n = self.board.point(point)
            human = n > 0
            count = abs(n)
            if count == 0:
                continue
            shown = min(count, 5)
            for i in range(shown):
                label = str(count) if (i == shown - 1 and count > 5) else None
                cx, cy = self.point_center(g, point, i)
                self._disc(p, cx, cy, g["r"], human, label)
            if point in self.source_points or point == self.carrying:
                cx, cy = self.point_center(g, point, shown - 1)
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(SRC_RING, 6 if point == self.carrying else 3))
                p.drawEllipse(QPointF(cx, cy), g["r"] + 3, g["r"] + 3)

    def _draw_bar(self, p, g):
        for side, human in ((0, True), (1, False)):
            for i in range(self.board.bar(side)):
                cx, cy = self.point_center(g, BAR, i, side=side)
                self._disc(p, cx, cy, g["r"], human)
        if BAR in self.source_points or BAR == self.carrying:
            bx0, bx1 = g["bar"]
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(SRC_RING, 6 if BAR == self.carrying else 3))
            p.drawRect(QRectF(bx0 + 2, g["H"] / 2 - 34, bx1 - bx0 - 4, 68))

    def _draw_off(self, p, g):
        ox0, ox1 = g["off"]
        p.setBrush(QBrush(FRAME))
        p.setPen(Qt.NoPen)
        p.drawRect(QRectF(ox0, g["margin"], ox1 - ox0, g["H"] - 2 * g["margin"]))
        bar_h = 11
        for side, human, top in ((0, True, False), (1, False, True)):
            for i in range(self.board.off(side)):
                y = (g["margin"] + 4 + i * (bar_h + 2)) if top else (
                    g["H"] - g["margin"] - bar_h - 4 - i * (bar_h + 2))
                p.setBrush(QBrush(HUMAN if human else ENGINE))
                p.setPen(QPen(HUMAN_EDGE if human else ENGINE_EDGE, 1))
                p.drawRect(QRectF(ox0 + 4, y, ox1 - ox0 - 8, bar_h))
        if OFF in self.dest_points:
            p.setBrush(QBrush(DEST_FILL))
            p.setPen(Qt.NoPen)
            p.drawRect(QRectF(ox0, g["H"] / 2, ox1 - ox0, g["H"] / 2 - g["margin"]))

    def _draw_cube(self, p, g):
        box = g["cube_rect"]
        r = box.width() / 2
        p.setBrush(QBrush(QColor("#f5f0e1")))
        p.setPen(QPen(QColor("#2b2b2b"), 2))
        p.drawRoundedRect(box, 6, 6)
        p.setPen(QPen(QColor("#222")))
        p.setFont(QFont("Arial", int(r * 0.85), QFont.Bold))
        p.drawText(box, Qt.AlignCenter, str(self.cube_value))
        if self.wink_cube and self.wink_on:    # pulse ring when the engine doubles
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(SRC_RING, 4))
            p.drawRoundedRect(box.adjusted(-5, -5, 5, 5), 9, 9)

    def _pip_face(self, p, x, y, s, value, face=None, border=None):
        p.setBrush(QBrush(face if face is not None else QColor("#fafafa")))
        p.setPen(QPen(border if border is not None else QColor("#222"), 2))
        p.drawRoundedRect(QRectF(x, y, s, s), 6, 6)
        p.setBrush(QBrush(QColor("#222")))
        p.setPen(Qt.NoPen)
        r = s * 0.09
        q = s / 4.0
        pat = {
            1: [(2, 2)], 2: [(1, 1), (3, 3)], 3: [(1, 1), (2, 2), (3, 3)],
            4: [(1, 1), (1, 3), (3, 1), (3, 3)],
            5: [(1, 1), (1, 3), (2, 2), (3, 1), (3, 3)],
            6: [(1, 1), (1, 2), (1, 3), (3, 1), (3, 2), (3, 3)],
        }[value]
        for gx, gy in pat:
            p.drawEllipse(QPointF(x + gx * q, y + gy * q), r, r)

    def _draw_dice(self, p, g):
        s = g["dice_s"]
        gap = 12
        y = g["H"] / 2 - s / 2
        if self.opening and self.open_dice:    # opening roll: one die each, colour-coded
            d_you, d_eng = self.open_dice
            og = 34
            x = g["W"] * 0.70 - (2 * s + og) / 2
            xe = x + s + og
            self._pip_face(p, x, y, s, d_you, face=QColor("#f2ead6"), border=QColor("#c8a24a"))
            self._pip_face(p, xe, y, s, d_eng, face=QColor("#f6e7e7"), border=QColor("#9e2b25"))
            p.setFont(QFont("Arial", 10, QFont.Bold))
            p.setPen(QPen(QColor("#e9d9a8")))
            p.drawText(QRectF(x, y - 22, s, 18), Qt.AlignCenter, "You")
            p.setPen(QPen(QColor("#e6a6a0")))
            p.drawText(QRectF(xe, y - 22, s, 18), Qt.AlignCenter, "Engine")
            p.setBrush(Qt.NoBrush)
            if self.open_resolved:
                win_x = x if d_you > d_eng else xe  # ring the higher die (starts)
                p.setPen(QPen(SRC_RING, 3))
                p.drawRoundedRect(QRectF(win_x - 4, y - 4, s + 8, s + 8), 8, 8)
            elif not self.open_rolling and self.wink_on:
                # Still winding, waiting to be clicked — pulse to say so. Once
                # thrown, stop: a pulsing tie would read as "click me again".
                p.setPen(QPen(SRC_RING, 3))
                p.drawRoundedRect(QRectF(x - 10, y - 10, (xe - x) + s + 20, s + 20), 10, 10)
            return
        if self.dice:
            total = len(self.dice) * s + (len(self.dice) - 1) * gap
            x0 = g["W"] * 0.70 - total / 2
            x = x0
            for v in self.dice:
                self._pip_face(p, x, y, s, v)
                x += s + gap
            if self.wink_dice and self.wink_on:
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(SRC_RING, 3))
                p.drawRoundedRect(QRectF(x0 - 6, y - 6, total + 12, s + 12), 9, 9)
            return
        if self.wink_dice:                     # first roll (no previous) — faint, static placeholder
            total = 2 * s + gap
            x = g["W"] * 0.70 - total / 2
            for _ in range(2):
                p.setBrush(QBrush(QColor(250, 250, 250, 40)))
                p.setPen(QPen(QColor(255, 210, 31, 90), 2))
                p.drawRoundedRect(QRectF(x, y, s, s), 6, 6)
                x += s + gap

    def _draw_numbers(self, p, g):
        """Both point numberings in the frame, so any notation can be found on the
        board.

        Backgammon has no single numbering: each player counts 1-24 from their own
        home, and moves are always written from the mover's own view. So your
        moves and hints read off the ivory numbers, and the engine's log lines
        read off the red ones. The two always sum to 25.
        """
        m, pw = g["margin"], g["pw"]
        for point in range(1, 25):
            col, is_top = self._col_row(point)
            x0 = g["xL"][col]
            y = 1 if is_top else g["H"] - m + 1
            p.setFont(QFont("Arial", 8, QFont.Bold))
            p.setPen(QPen(HUMAN_NUM))
            p.drawText(QRectF(x0, y, pw / 2 - 1, m - 2),
                       Qt.AlignRight | Qt.AlignVCenter, str(point))
            p.setFont(QFont("Arial", 7))
            p.setPen(QPen(ENGINE_NUM))
            p.drawText(QRectF(x0 + pw / 2 + 2, y, pw / 2 - 2, m - 2),
                       Qt.AlignLeft | Qt.AlignVCenter, str(25 - point))

    def _draw_pips(self, p, g):
        p.setPen(QPen(QColor("#eafff2")))
        p.setFont(QFont("Arial", 11, QFont.Bold))
        p.drawText(QRectF(g["W"] * 0.10, g["H"] - g["margin"] - 22, 220, 20),
                   Qt.AlignLeft, f"You: {self.board.pip_count(0)} pips")
        p.drawText(QRectF(g["W"] * 0.10, g["margin"] + 2, 220, 20),
                   Qt.AlignLeft, f"Engine: {self.board.pip_count(1)} pips")

    # --- hover boxes ---
    # Anything the board wants you to click pulses; hovering it spells out what
    # the click will do, as small boxes that vanish when you move away.
    def _zone_actions(self, zone):
        """The (label, key) buttons a zone offers right now — empty if none."""
        if zone == "dice":
            if self.opening:
                # The opening dice are winding; any click rolls them, so a box
                # offering a second click would only get in the way.
                return []
            if self.wink_dice:                 # your turn, nothing rolled yet
                return [("Roll dice", "roll")]
        elif zone == "cube":
            if self.wink_cube:                 # engine doubled — you must answer
                return [("Accept", "accept"), ("Fold", "fold")]
            if self.can_double:
                return [("Double", "double")]
        return []

    def _zone_anchor(self, zone, g):
        return g["dice_rect"] if zone == "dice" else g["cube_rect"]

    def _zone_boxes(self, zone, g):
        """[(rect, label, key)] laid out beside the zone's anchor."""
        acts = self._zone_actions(zone)
        if not acts:
            return []
        w, h, gap = 92.0, 28.0, 6.0
        anchor = self._zone_anchor(zone, g)
        out = []
        if zone == "dice":                     # below the dice, side by side
            total = len(acts) * w + (len(acts) - 1) * gap
            x = anchor.center().x() - total / 2
            y = anchor.bottom() + 12
            for label, key in acts:
                out.append((QRectF(x, y, w, h), label, key))
                x += w + gap
        else:                                  # cube hugs the left edge -> stack right
            x = anchor.right() + 10
            total = len(acts) * h + (len(acts) - 1) * gap
            y = anchor.center().y() - total / 2
            for label, key in acts:
                out.append((QRectF(x, y, w, h), label, key))
                y += h + gap
        return out

    def _zone_at(self, pos):
        """Which zone the cursor is in — counting its boxes, so moving onto a box
        doesn't dismiss the very box you're reaching for."""
        g = self._geom
        if g is None:
            return None
        for zone in ("cube", "dice"):
            if not self._zone_actions(zone):
                continue
            hot = [self._zone_anchor(zone, g)] + [r for r, _, _ in self._zone_boxes(zone, g)]
            if any(r.adjusted(-8, -8, 8, 8).contains(pos) for r in hot):
                return zone
        return None

    def _panel(self, p, g, title, w, rows_h):
        """Frame a centred overlay panel; returns the y to start drawing rows at."""
        pad = 18
        h = pad * 2 + 30 + rows_h
        x = (g["W"] - w) / 2
        y = (g["H"] - h) / 2
        p.setBrush(QBrush(QColor(16, 18, 16, 242)))
        p.setPen(QPen(SRC_RING, 2))
        p.drawRoundedRect(QRectF(x, y, w, h), 10, 10)
        p.setPen(QPen(QColor("#f7f2e2")))
        p.setFont(QFont("Arial", 12, QFont.Bold))
        p.drawText(QRectF(x + pad, y + pad - 2, w - 2 * pad, 24), Qt.AlignLeft, title)
        return x, y + pad + 26

    def _draw_hint(self, p, g):
        """The ranked moves, best first, with their equities."""
        if not self.hint_rows:
            return
        pad, lh, w = 18, 24, 330
        x, ty = self._panel(p, g, "Hint — best moves", w, lh * len(self.hint_rows))
        for i, (notation, eq) in enumerate(self.hint_rows):
            best = i == 0
            p.setFont(QFont("Arial", 10, QFont.Bold if best else QFont.Normal))
            p.setPen(QPen(SRC_RING if best else QColor("#ded8c6")))
            p.drawText(QRectF(x + pad, ty, w - 2 * pad - 60, lh),
                       Qt.AlignLeft | Qt.AlignVCenter, notation or "(dance)")
            p.drawText(QRectF(x + w - pad - 60, ty, 60, lh),
                       Qt.AlignRight | Qt.AlignVCenter, f"{eq:+.2f}")
            ty += lh

    def _draw_help(self, p, g):
        """A panel over the board listing what everything does."""
        if not self.show_help:
            return
        pad, lh, key_w = 18, 22, 78
        w = 470
        x, ty = self._panel(p, g, "How to play", w, lh * len(HELP_LINES))
        for key, text in HELP_LINES:
            if key is None:                      # full-width intro line
                p.setFont(QFont("Arial", 9))
                p.setPen(QPen(QColor("#b9b1a0")))
                p.drawText(QRectF(x + pad, ty, w - 2 * pad, lh),
                           Qt.AlignLeft | Qt.AlignVCenter, text)
                ty += lh
                continue
            p.setFont(QFont("Arial", 9, QFont.Bold))
            p.setPen(QPen(SRC_RING))
            p.drawText(QRectF(x + pad, ty, key_w, lh), Qt.AlignLeft | Qt.AlignVCenter, key)
            p.setFont(QFont("Arial", 9))
            p.setPen(QPen(QColor("#ded8c6")))
            p.drawText(QRectF(x + pad + key_w, ty, w - 2 * pad - key_w, lh),
                       Qt.AlignLeft | Qt.AlignVCenter, text)
            ty += lh

    def _draw_hover(self, p, g):
        if self.hover_zone is None:
            return
        p.setFont(QFont("Arial", 10, QFont.Bold))
        for rect, label, _key in self._zone_boxes(self.hover_zone, g):
            p.setBrush(QBrush(QColor(20, 20, 20, 225)))
            p.setPen(QPen(SRC_RING, 2))
            p.drawRoundedRect(rect, 6, 6)
            p.setPen(QPen(QColor("#f7f2e2")))
            p.drawText(rect, Qt.AlignCenter, label)

    # --- interaction ---
    def hit_test(self, x, y):
        g = self._geom
        if g is None:
            return None
        H = g["H"]
        ox0, ox1 = g["off"]
        if ox0 <= x <= ox1:
            return OFF if y > H / 2 else None
        bx0, bx1 = g["bar"]
        if bx0 <= x <= bx1:
            return BAR
        for c in range(12):
            if g["xL"][c] <= x < g["xL"][c] + g["pw"]:
                return (c + 13) if y < H / 2 else (12 - c)
        return None

    def _set_hover(self, zone):
        if zone != self.hover_zone:
            self.hover_zone = zone
            self.update()

    def mouseMoveEvent(self, ev):
        self._set_hover(self._zone_at(ev.position()))

    def leaveEvent(self, ev):
        self._set_hover(None)

    def mousePressEvent(self, ev):
        pos = ev.position()
        if ev.button() == Qt.RightButton:
            self.on_click(None, "right")
            return
        g = self._geom
        # A visible hover box wins the click — it sits over the board.
        if g is not None and self.hover_zone is not None:
            for rect, _label, key in self._zone_boxes(self.hover_zone, g):
                if rect.contains(pos):
                    self._set_hover(None)
                    self.on_action(key)
                    return
        if g is not None and g["dice_rect"].contains(pos):
            self.on_dice()
            return
        self.on_click(self.hit_test(pos.x(), pos.y()), "left")


class EvalBar(QWidget):
    """Vertical evaluation bar from your (bottom) point of view: the ivory fill
    rises from the bottom with your win probability; red (engine) fills the top."""

    def __init__(self):
        super().__init__()
        self.setFixedWidth(46)
        self.human_wp = 0.5

    def set_value(self, wp):
        self.human_wp = max(0.0, min(1.0, wp))
        self.update()

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        pad = 6
        x0, y0, bw, bh = pad, pad, w - 2 * pad, h - 2 * pad
        p.fillRect(self.rect(), FRAME)
        split = y0 + bh * (1.0 - self.human_wp)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(ENGINE))
        p.drawRect(QRectF(x0, y0, bw, split - y0))     # engine (top)
        p.setBrush(QBrush(HUMAN))
        p.drawRect(QRectF(x0, split, bw, y0 + bh - split))  # you (bottom)
        p.setPen(QPen(QColor("#cfcfcf"), 1))            # 50% line
        p.drawLine(int(x0), int(y0 + bh / 2), int(x0 + bw), int(y0 + bh / 2))
        p.setPen(QPen(QColor("#2b2b2b")))
        p.setFont(QFont("Arial", 9, QFont.Bold))
        p.drawText(QRectF(x0, y0 + bh - 18, bw, 16), Qt.AlignCenter,
                   f"{self.human_wp * 100:.0f}%")
        p.drawText(QRectF(x0, y0 + 2, bw, 16), Qt.AlignCenter,
                   f"{(1 - self.human_wp) * 100:.0f}%")
        p.end()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Backgammon — bgcore")
        # Volume persists between sessions — nobody wants to re-mute every launch.
        # First run starts at half.
        self.settings = QSettings("ChrisWhittington", "Backgammon-NN")
        vol = float(self.settings.value("sound/volume", 0.5))
        self.sfx = Sfx(vol)

        file_menu = self.menuBar().addMenu("&File")
        new_act = QAction("&New Game", self)
        new_act.setShortcut("Ctrl+N")
        new_act.triggered.connect(self.new_game)
        undo_act = QAction("&Undo move", self)
        undo_act.setShortcut("Ctrl+Z")
        undo_act.triggered.connect(self.undo_submove)
        quit_act = QAction("&Quit", self)
        quit_act.setShortcut("Ctrl+Q")
        quit_act.triggered.connect(self.close)
        file_menu.addAction(new_act)
        file_menu.addAction(undo_act)
        file_menu.addSeparator()
        file_menu.addAction(quit_act)

        onnx_path = ROOT / "models" / "td.onnx"
        ckpt = ROOT / "models" / "td_latest.pt"
        self.opponents = {}
        self.evaluator = None
        neural1 = None

        # Prefer the native (Rust/ONNX) net: same play, several times faster, and
        # no torch — it's the only neural path in the packaged app. Fall back to
        # the torch checkpoint for source checkouts built without the onnx
        # feature.
        neurals = []
        if hasattr(bgcore, "Neural") and onnx_path.exists():
            neurals = [NativeNeuralEngine(onnx_path, lookahead=p) for p in (0, 1, 2)]
        elif ckpt.exists():
            neural0 = NeuralEngine(ckpt, lookahead=0)
            neurals = [
                neural0,
                NeuralEngine(ckpt, lookahead=1, share=neural0),
                NeuralEngine(ckpt, lookahead=2, share=neural0),
            ]
        if neurals:
            for e in neurals:
                self.opponents[e.name] = e
            self.evaluator = neurals[0]

        # Phase-routing engines: champion for contact, race net for race (the race
        # net beats min-pip by ~+0.25 PPG in the bear-off). PHASE_CONTACT picks the
        # contact net — the mature champion `td.onnx` by default; swap to
        # `td_contact.onnx` for the fresh co-trained contact net (plays within noise
        # of the champion at contact).
        PHASE_CONTACT = "td.onnx"
        race_onnx = ROOT / "models" / "td_race.onnx"
        contact_onnx = ROOT / "models" / PHASE_CONTACT
        if (hasattr(bgcore, "PhaseNeural") and race_onnx.exists()
                and contact_onnx.exists()):
            for p in (0, 1, 2):
                e = NativePhaseEngine(contact_onnx, race_onnx, lookahead=p)
                self.opponents[e.name] = e

        self._cube_ro = None
        rollout = None
        if hasattr(bgcore, "Rollouts") and onnx_path.exists():
            rollout = RolloutEngine(onnx_path, movetime_ms=800, truncate_plies=9, candidates=5)
            self.opponents[rollout.name] = rollout
            # Rollout evaluator for cube decisions — movetime-budgeted, no filter.
            self._cube_ro = bgcore.Rollouts(str(onnx_path), 0, 9, 0, 7, 400, 0)
        self.opponents["HCE (heuristic)"] = HceEngine()
        self.opponents["Random"] = RandomEngine()

        # Play the strongest opponent available by default: rollouts beat 2-ply
        # search, which beats shallower search (see the Elo ladder in the README).
        # `neurals` is ordered by depth, so its last entry is the deepest.
        best_neural = neurals[-1] if neurals else None
        self.default_engine = (
            rollout or best_neural or self.opponents["HCE (heuristic)"]
        )
        # Hints list the *top few* moves, so they need an engine that ranks every
        # move — RolloutEngine only ever reports the one it picked. So hints use
        # the deepest search instead of the outright strongest engine.
        self.hint_engine = best_neural or self.opponents["HCE (heuristic)"]

        self.view = BoardView(self.on_click, self.on_dice, self.on_hover_action)
        self.roll_btn = QPushButton("Roll")
        self.double_btn = QPushButton("Double")
        self.take_btn = QPushButton("Take")
        self.drop_btn = QPushButton("Drop")
        self.help_btn = HoverButton("?", self.on_help_hover)
        self.help_btn.setToolTip("Hover for how to play")
        self.help_btn.setFixedWidth(30)
        self._hint_key = None
        self._hint_cache = None
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.setToolTip("Take back the last checker you moved (Ctrl+Z)")
        self.hint_btn = HoverButton("Hint", self.on_hint_hover)
        self.hint_btn.setToolTip("Hover to see the best moves ranked")
        self.new_btn = QPushButton("New Game")
        self.opp_box = QComboBox()
        self.opp_box.addItems(list(self.opponents.keys()))
        self.opp_box.setCurrentText(self.default_engine.name)
        self.roll_btn.clicked.connect(self.on_dice)
        self.double_btn.clicked.connect(self.on_double)
        self.take_btn.clicked.connect(self.on_take)
        self.drop_btn.clicked.connect(self.on_drop)
        self.undo_btn.clicked.connect(self.undo_submove)
        self.hint_btn.clicked.connect(self.on_hint)
        self.new_btn.clicked.connect(self.new_game)
        self.take_btn.setVisible(False)
        self.drop_btn.setVisible(False)

        self.score_label = QLabel("You 0 — 0 Engine")
        self.score_label.setStyleSheet("font-weight:bold; padding:0 8px;")

        self.vol_slider = QSlider(Qt.Horizontal)
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setValue(int(self.sfx.volume * 100))
        self.vol_slider.setFixedWidth(90)
        self.vol_slider.setToolTip("Sound effects volume (0 = silent)")
        self.vol_slider.valueChanged.connect(self.on_volume)
        self.vol_label = QLabel()
        self.vol_label.setToolTip("Sound effects volume (0 = silent)")

        controls = QHBoxLayout()
        for w in (self.roll_btn, self.double_btn, self.take_btn, self.drop_btn,
                  self.undo_btn, self.hint_btn, self.new_btn,
                  QLabel("Opponent:"), self.opp_box):
            controls.addWidget(w)
        controls.addWidget(self.help_btn)
        controls.addStretch(1)
        controls.addWidget(self.vol_label)
        controls.addWidget(self.vol_slider)
        controls.addWidget(self.score_label)
        self._sync_vol_label()

        self.moves = QListWidget()
        self.moves.setFixedWidth(230)
        self.moves.setStyleSheet("font-family: Consolas, monospace; font-size: 12px;")

        self.eval_bar = EvalBar()

        board_row = QHBoxLayout()
        board_row.addWidget(self.view, 1)
        board_row.addWidget(self.eval_bar)
        board_row.addWidget(self.moves)

        self.status = QLabel("")
        self.status.setStyleSheet("padding:6px; font-size:14px;")

        central = QWidget()
        lay = QVBoxLayout(central)
        lay.addLayout(controls)
        lay.addLayout(board_row, 1)
        lay.addWidget(self.status)
        self.setCentralWidget(central)

        self.rng = random.Random()
        self._roll_timer = QTimer(self)
        self._roll_timer.timeout.connect(self._roll_frame)
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._anim_frame)
        self._wink_timer = QTimer(self)
        self._wink_timer.setInterval(430)
        self._wink_timer.timeout.connect(self._wink_tick)
        # Opening dice count 1-6 together until you click to roll.
        self._open_timer = QTimer(self)
        self._open_timer.setInterval(250)
        self._open_timer.timeout.connect(self._open_tick)
        self.busy = False
        self.score = [0, 0]          # cumulative points [you, engine]
        self.pending_double = False  # engine offered a double, awaiting take/drop
        self.combos = {}             # two-dice destinations for the held checker
        self._tasks = []             # in-flight engine work
        self._gen = 0               # bumped when in-flight results stop being valid
        self.new_game()

    @property
    def opponent(self):
        return self.opponents[self.opp_box.currentText()]

    def _pos_eval(self, board):
        if self.evaluator is not None:
            return self.evaluator.static_equity(board)
        return bgcore.hce_equity(board)

    def _cube_eval(self, board):
        """Equity for cube decisions — rollout-based when available, else static."""
        if self._cube_ro is not None:
            return self._cube_ro.equity(board)
        return self._pos_eval(board)

    def _win_prob(self):
        """Your (bottom side's) win probability for the eval bar."""
        disp = self.board if self.human_turn else self.board.swap_perspective()
        if self.evaluator is not None:
            return self.evaluator.win_prob(disp)
        import math
        return 1.0 / (1.0 + math.exp(-1.2 * bgcore.hce_equity(disp)))

    def _wink_tick(self):
        self.view.wink_on = not self.view.wink_on
        self.view.update()

    # --- view sync ---
    def refresh(self, message=None):
        disp = self.board if self.human_turn else self.board.swap_perspective()
        self.view.board = disp
        roll_time = (self.human_turn and not self.remaining and not self.game_over
                     and not self.busy and not self.pending_double and not self.opening)
        if self._roll_timer.isActive():
            pass                               # a tumble owns the dice; don't spoil it
        elif self.remaining:
            self.view.dice = list(self.remaining)
        else:
            self.view.dice = list(self.roll)   # roll-time keeps the previous roll (it winks)
        self.view.carrying = self.carrying
        self.view.source_points = (
            {s[0] for s in self.subs} if (self.human_turn and self.remaining) else set())
        # Light up single-die destinations plus anywhere two dice can reach.
        self.combos = self._combo_dests()
        self.view.dest_points = (
            ({s[1] for s in self.subs if s[0] == self.carrying} | set(self.combos))
            if self.carrying is not None else set())
        self.view.floating = None
        self.view.cube_value = self.cube_value
        self.view.cube_owner = self.cube_owner
        self.view.opening = self.opening
        # While the opening dice wind, _open_tick owns the faces — don't fight it.
        self.view.open_dice = self._open if self.opening else None
        self.view.open_resolved = self.opening and self.open_resolved
        self.view.open_rolling = self.opening and self.open_rolling
        if message is not None:
            self.status.setText(message)
        ready = (self.human_turn and not self.remaining and not self.game_over
                 and not self.busy and not self.opening)
        self.roll_btn.setEnabled(ready and not self.pending_double)
        self.double_btn.setEnabled(ready and not self.pending_double and self.may_double(0))
        self.take_btn.setVisible(self.pending_double)
        self.drop_btn.setVisible(self.pending_double)
        self.undo_btn.setEnabled(
            bool(self.undo_stack) and self.human_turn and not self.busy and not self.game_over)
        self.hint_btn.setEnabled(
            self.human_turn and bool(self.remaining) and self.full_roll and not self.game_over)
        self.score_label.setText(f"You {self.score[0]} — {self.score[1]} Engine")
        self.eval_bar.set_value(self._win_prob())
        self.view.wink_dice = roll_time
        self.view.wink_cube = self.pending_double
        self.view.can_double = ready and not self.pending_double and self.may_double(0)
        # The opening dice pulse too, until they're clicked.
        if self.view.wink_dice or self.view.wink_cube or self.opening:
            if not self._wink_timer.isActive():
                self.view.wink_on = True
                self._wink_timer.start()
        elif self._wink_timer.isActive():
            self._wink_timer.stop()
            self.view.wink_on = True
        # Whatever the cursor sits on may now offer different actions.
        self.view.hover_zone = self.view._zone_at(
            self.view.mapFromGlobal(QCursor.pos()).toPointF())
        self.view.update()

    # --- game flow ---
    def new_game(self):
        self._roll_timer.stop()
        self._anim_timer.stop()
        # Anything the engine is still thinking about belongs to the old game.
        self._gen += 1
        self.board = bgcore.Board.starting()
        self.human_turn = True
        self.game_over = False
        self.busy = False
        self.remaining: list[int] = []
        self.roll = ()
        self.subs = []
        self.carrying = None
        self.full_roll = False
        self.human_steps = []
        self.undo_stack = []
        self.moves.clear()
        self.turn_no = 1
        self.cube_value = 1
        self.cube_owner = None       # None = centered, 0 = you, 1 = engine
        self.pending_double = False
        self.take_btn.setVisible(False)
        self.drop_btn.setVisible(False)
        # Opening roll — one die each, the higher starts. The dice wind through
        # 1-6 until you click them; the click is what actually rolls.
        self.opening = True
        self.open_resolved = False
        self.open_rolling = False
        self._open = (1, 4)
        self._open_winner = None
        self._open_k = 0
        self.view.opening = True
        self.view.open_resolved = False
        self.view.open_rolling = False
        self.view.open_dice = self._open
        self._open_timer.start()
        self.refresh("Click the dice to roll for who starts.")

    def _open_tick(self):
        """Count both dice up 1-6 together, over and over, until the click."""
        self._open_k += 1
        v = (self._open_k % 6) + 1
        self._open = (v, v)
        self.view.open_dice = self._open
        self.view.update()

    def _open_roll(self):
        """The click that rolls: stop winding and throw one die each."""
        if not self.opening or self.open_rolling:
            return
        self.open_rolling = True
        self._open_timer.stop()
        self._open_throw()

    def _open_throw(self):
        """One throw of the opening dice. The higher die starts; a tie is shown
        and thrown again, as at the board — rather than quietly re-rolled until
        it happens to differ."""
        if not self.opening:
            return
        d_you, d_eng = self.rng.randint(1, 6), self.rng.randint(1, 6)
        self._open = (d_you, d_eng)
        self.view.open_dice = self._open
        self.sfx.play_dice()
        if d_you == d_eng:
            self.open_resolved = False
            self.refresh(f"Both threw {d_you} — throwing again.")
            QTimer.singleShot(950, self._open_throw)
            return
        self._open_winner = 0 if d_you > d_eng else 1
        self.open_resolved = True
        starter = "You" if self._open_winner == 0 else "Engine"
        self.refresh(f"Opening roll — You {d_you}, Engine {d_eng}. {starter} start.")
        QTimer.singleShot(1100, self._open_finish)

    def _open_finish(self):
        if not self.opening:
            return
        self._open_timer.stop()
        if self._open_winner is None:   # forced past without rolling (tests)
            self._open_winner = 0
        self.opening = False
        self.open_resolved = False
        self.open_rolling = False
        self.view.opening = False
        self.view.open_resolved = False
        self.view.open_rolling = False
        self.view.open_dice = None
        # The opening throw *is* the winner's first roll — they play those two
        # dice rather than rolling again. It also means no double can be offered
        # before the first move: the roll has already been made.
        d1, d2 = self._open
        while d1 == d2:
            # A real opening throw never ties (ties are re-thrown), so this only
            # bites when something skipped the throw and left the counter's face.
            d1, d2 = self.rng.randint(1, 6), self.rng.randint(1, 6)
        if self._open_winner == 0:                 # you start, on those dice
            self.human_turn = True
            self._begin_human_roll(d1, d2, lead="You start with")
        else:                                      # engine starts, on those dice
            self.human_turn = False
            self.busy = True
            self.roll = (d1, d2)
            self.refresh(f"Engine starts with {d1}-{d2}…")
            QTimer.singleShot(650, lambda: self.engine_play((d1, d2)))

    def _run_async(self, fn, then):
        """Run `fn` on a worker thread; call `then(result)` back on the UI thread.

        Results from a superseded game (New Game mid-think) are dropped: `gen` is
        bumped whenever the position they were computed for stops being current.

        `sync_engine` (set by the headless test) runs `fn` inline instead — the
        worker exists only to keep the real app's window responsive, and driving a
        cross-thread signal by hand from a test races the pump.
        """
        if getattr(self, "sync_engine", False):
            try:
                result = fn()
            except Exception as exc:  # noqa: BLE001 — mirror _on_task_done
                self.busy = False
                self.refresh(f"Engine error: {exc}")
                return
            if not self.game_over:
                then(result)
            return
        task = _Task(fn, self._gen)
        task.signals.done.connect(self._on_task_done)
        # Hold the task: QThreadPool deletes the runnable after run(), and the
        # queued signal must still have its sender alive to be delivered.
        self._tasks.append((task, then))
        QThreadPool.globalInstance().start(task)

    def _on_task_done(self, result, gen):
        entry = next((t for t in self._tasks if t[0].signals is self.sender()), None)
        if entry is not None:
            self._tasks.remove(entry)
        if gen != self._gen or self.game_over:
            return                      # a new game started while this was thinking
        if isinstance(result, Exception):
            self.busy = False
            self.refresh(f"Engine error: {result}")
            return
        entry[1](result)

    def _combo_dests(self):
        """Where the selected checker can reach using *two* dice, as
        `{final_point: first_submove}`.

        Skips any route whose intermediate landing hits a blot: a hit is a real
        event you should choose deliberately, not something to slide through on
        the way somewhere else.
        """
        combos = {}
        if self.carrying is None or not self.human_turn:
            return combos
        for sub in self.subs:
            frm, to, die, result = sub
            if frm != self.carrying or to == OFF:
                continue
            if self.board.point(to) == -1:      # a lone opponent checker: a hit
                continue
            rest = list(self.remaining)
            rest.remove(die)
            if not rest:
                continue
            for nxt in bgcore.submoves(result, rest):
                if nxt[0] == to and nxt[1] != OFF:
                    combos.setdefault(nxt[1], sub)
        return combos

    def may_double(self, side):
        return (not self.game_over and self.cube_value < 64
                and self.cube_owner in (None, side))

    def closeEvent(self, ev):
        self.sfx.stop_all()
        super().closeEvent(ev)

    def on_help_hover(self, showing):
        self.view.show_help = showing
        self.view.update()

    def _sync_vol_label(self):
        self.vol_label.setText("🔇" if self.sfx.volume <= 0 else "🔊")

    def on_volume(self, pct):
        self.sfx.set_volume(pct / 100.0)
        self.settings.setValue("sound/volume", self.sfx.volume)
        self._sync_vol_label()

    def undo_submove(self):
        """Take back the last checker you moved this turn, restoring its die.

        Only within your own turn: once the turn commits the engine has replied
        and there's nothing to take back to.
        """
        if not self.undo_stack or not self.human_turn or self.busy or self.game_over:
            return
        self.board, self.remaining, self.human_steps, self.full_roll = self.undo_stack.pop()
        self.carrying = None
        self.subs = bgcore.submoves(self.board, self.remaining)
        left = len(self.undo_stack)
        self.refresh("Took back a checker."
                     + ("" if left else " Back to the start of your turn."))

    def on_hover_action(self, key):
        """A hover box on the board was clicked — same actions as the buttons."""
        {
            "roll": self.on_dice,
            "double": self.on_double,
            "accept": self.on_take,
            "fold": self.on_drop,
        }[key]()

    def on_dice(self):
        if self.opening:                # the click that rolls for who starts
            self._open_roll()
            return
        if not (self.human_turn and not self.remaining and not self.game_over and not self.busy):
            return
        self.busy = True
        self._wink_timer.stop()         # stop the roll prompt once rolling starts
        self.view.wink_dice = False
        d1, d2 = self.rng.randint(1, 6), self.rng.randint(1, 6)
        self._tumble(d1, d2, lambda: self._begin_human_roll(d1, d2))

    def _tumble(self, d1, d2, then):
        """Tumble the dice briefly, land on `d1, d2`, then call `then`.

        Both sides roll through here, so the engine's roll is seen and heard
        rather than just appearing. Kept short — it sits between every move.
        """
        self.sfx.play_dice()
        self._roll_final = (d1, d2)
        self._roll_then = then
        self._roll_frames = 8
        self._roll_timer.start(45)

    def _roll_frame(self):
        self._roll_frames -= 1
        if self._roll_frames > 0:
            self.view.dice = [self.rng.randint(1, 6), self.rng.randint(1, 6)]
            self.view.update()
            return
        self._roll_timer.stop()
        d1, d2 = self._roll_final
        self.view.dice = [d1, d2]
        # Paint the landed dice *now*, not when the event loop next gets a turn:
        # `then` can block for the best part of a second (the engine choosing via
        # rollouts), and a queued update() wouldn't run until after it — leaving
        # the last tumble frame on screen looking like the roll, then flicking to
        # the real one as the engine moves.
        self.view.repaint()
        self._roll_then()

    def _begin_human_roll(self, d1, d2, lead="Rolled"):
        """Start your turn on dice `d1, d2` — from a fresh roll, or from the
        opening throw, which *is* the winner's first roll."""
        self.roll = (d1, d2)
        self.remaining = [d1] * 4 if d1 == d2 else [d1, d2]
        self.full_roll = True
        self.carrying = None
        self.human_steps = []
        self.undo_stack = []
        self.busy = False
        self.subs = bgcore.submoves(self.board, self.remaining)
        if not self.subs:
            self.refresh(f"{lead} {d1}-{d2}: no legal move (dance).")
            QTimer.singleShot(700, self.end_human_turn)
            return
        self.refresh(f"{lead} {d1}-{d2}. Click a checker, then its destination.")

    def on_click(self, pid, button):
        if self.opening:                # any click on the board rolls
            self._open_roll()
            return
        if not self.human_turn or self.game_over or self.busy or not self.remaining:
            return
        if button == "right":
            if self.carrying is not None:
                self.carrying = None
                self.refresh("Selection cleared. Click a checker.")
            return
        sources = {s[0] for s in self.subs}
        if self.carrying is None:
            if pid in sources:
                self.carrying = pid                    # select the source
                self.refresh("Now click a highlighted destination.")
            return
        if pid == self.carrying:
            self.carrying = None                       # click again to deselect
            self.refresh("Selection cleared. Click a checker.")
            return
        match = next((s for s in self.subs if s[0] == self.carrying and s[1] == pid), None)
        if match:
            self.apply_submove(match)
        elif pid in self.combos:
            # A two-dice destination: play the first leg, then the leg that lands
            # on the point you actually clicked. Each leg snapshots separately,
            # so Undo still takes them back one checker at a time.
            first = self.combos[pid]
            self.apply_submove(first)
            nxt = next((s for s in self.subs if s[0] == first[1] and s[1] == pid), None)
            if nxt is not None:
                self.apply_submove(nxt)
        elif pid in sources:
            self.carrying = pid                        # switch to a different source
            self.refresh()
        else:
            self.carrying = None                       # clicked elsewhere -> deselect
            self.refresh()

    def apply_submove(self, sub):
        frm, to, die, result = sub
        # Snapshot first, so Undo can put this checker back. Boards are
        # immutable, and the lists are small — a plain stack is enough.
        self.undo_stack.append(
            (self.board, list(self.remaining), list(self.human_steps), self.full_roll))
        self.board = result
        self.remaining.remove(die)
        self.full_roll = False
        self.carrying = None
        self.human_steps.append((frm, to, die))
        self.sfx.play_place()
        pts = self.board.winner_points()
        if pts is not None and pts > 0:
            self._log_move("You", self.roll, self.human_steps, float(pts))
            self._end_game(0, self.cube_value * pts, f"You win {self._pts_name(pts)}")
            return
        self.subs = bgcore.submoves(self.board, self.remaining)
        if not self.subs:
            self.end_human_turn()
        else:
            self.refresh("Continue your move.")

    def end_human_turn(self):
        if self.human_steps:
            eq = -self._pos_eval(self.board.swap_perspective())
            self._log_move("You", self.roll, self.human_steps, eq)
        # The turn is committed and the engine is about to reply — past here
        # there's nothing to take back to.
        self.undo_stack.clear()
        self.human_turn = False
        self.board = self.board.swap_perspective()
        self.remaining = []
        self.carrying = None
        self.subs = []
        self.busy = True
        self.refresh("Engine thinking…")
        QTimer.singleShot(350, self.engine_turn)

    # --- doubling cube ---
    def on_double(self):
        if not (self.human_turn and not self.remaining and not self.game_over
                and not self.busy and not self.pending_double and not self.opening
                and self.may_double(0)):
            return
        eq = self._cube_eval(self.board)  # your equity (rollout-based when available)
        if should_take(eq):
            self.cube_value *= 2
            self.cube_owner = 1  # engine owns the cube now
            self.refresh(f"You double. Engine takes — cube is {self.cube_value}. Roll.")
        else:
            self._end_game(0, self.cube_value, "Engine drops")

    def on_take(self):
        if not self.pending_double:
            return
        self.cube_value *= 2
        self.cube_owner = 0  # you own the cube now
        self.pending_double = False
        self.refresh(f"You take — cube is {self.cube_value}. Engine plays…")
        QTimer.singleShot(300, self.engine_play)

    def on_drop(self):
        if not self.pending_double:
            return
        self.pending_double = False
        self._end_game(1, self.cube_value, "You drop")

    def _end_game(self, winner, points, reason):
        self.game_over = True
        self.busy = False
        self.pending_double = False
        self.score[winner] += points
        who = "You" if winner == 0 else "Engine"
        self.refresh(f"{reason}. {who} +{points} (score {self.score[0]}-{self.score[1]}). "
                     f"New Game to continue.")

    def engine_turn(self):
        """Engine's turn: consider doubling, otherwise roll and play."""
        if self.game_over:
            return
        if not self.may_double(1):
            self.engine_play()
            return
        # The cube decision is a rollout too — off the UI thread with the rest.
        board = self.board
        self.busy = True
        self._run_async(lambda: self._cube_eval(board), self._engine_cube_decided)

    def _engine_cube_decided(self, eq):
        if should_double(eq, True):
            self.pending_double = True
            self.busy = True
            self.refresh(f"Engine doubles to {self.cube_value * 2}! Take or Drop?")
            return
        self.engine_play()

    def engine_play(self, dice=None):
        """Play the engine's turn. `dice` forces the roll — used for the opening
        throw, which the engine plays rather than rolling afresh."""
        if self.game_over:
            return
        self.busy = True
        if dice is not None:
            # The opening throw is already on the table. Tumbling it here would
            # look exactly like the engine re-rolling dice it has already won.
            self.roll = dice
            self.view.dice = list(dice)
            self.view.update()
            self._engine_move()
            return
        d1, d2 = self.rng.randint(1, 6), self.rng.randint(1, 6)
        self.roll = (d1, d2)
        # Tumble first, then think: choosing can block for the best part of a
        # second (rollouts), and the dice should be seen landing before that.
        # Don't name the roll here — the dice haven't landed yet.
        self.refresh("Engine rolls…")
        self._tumble(d1, d2, self._engine_move)

    def _engine_move(self):
        if self.game_over:
            return
        d1, d2 = self.roll
        board, engine = self.board, self.opponent
        self.refresh(f"Engine thinking on {d1}-{d2}…")
        self._run_async(lambda: engine.choose(board, d1, d2), self._engine_chose)

    def _engine_chose(self, chosen):
        d1, d2 = self.roll
        nxt, pts, steps, eq = chosen
        self._engine_result = (nxt, pts, steps, eq, d1, d2)
        self._build_engine_animation(steps)
        if self._anim_queue:
            self.view.dice = [d1, d2]
            self._anim_step = 0
            self._start_anim_segment()
        else:
            self._finish_engine_move()

    def _build_engine_animation(self, steps):
        # Reconstruct intermediate engine-relative boards, then map each step to
        # display-space screen coordinates (engine point p -> display point 25-p).
        self._anim_queue = []
        g = self.view._geom
        if g is None:
            return
        cur = self.board
        boards = [cur]
        for f, t, _ in steps:
            cur = bgcore.apply_step(cur, f, t)
            boards.append(cur)
        for i, (f, t, _) in enumerate(steps):
            before, after = boards[i], boards[i + 1]
            dp_from = BAR if f == BAR else 25 - f
            dp_to = OFF if t == OFF else 25 - t
            fi = 0 if dp_from in (BAR, OFF) else max(abs(before.point(dp_from)) - 1, 0)
            ti = 0 if dp_to in (BAR, OFF) else max(abs(after.point(dp_to)) - 1, 0)
            self._anim_queue.append((
                before.swap_perspective(),
                self.view.point_center(g, dp_from, fi, side=1),
                self.view.point_center(g, dp_to, ti, side=1),
            ))

    def _start_anim_segment(self):
        disp_before, self._a_from, self._a_to = self._anim_queue[self._anim_step]
        self.view.board = disp_before
        self.view.source_points = set()
        self.view.dest_points = set()
        self._a_frame = 0
        self._a_frames = 24             # slower checker glide
        self._anim_timer.start(26)

    def _anim_frame(self):
        self._a_frame += 1
        u = self._a_frame / self._a_frames
        t = u * u * (3 - 2 * u)          # ease in-out
        x = self._a_from[0] + (self._a_to[0] - self._a_from[0]) * t
        y = self._a_from[1] + (self._a_to[1] - self._a_from[1]) * t
        self.view.floating = (x, y, False)
        self.view.update()
        if self._a_frame >= self._a_frames:
            self._anim_timer.stop()
            self.sfx.play_place()
            self._anim_step += 1
            if self._anim_step < len(self._anim_queue):
                self._start_anim_segment()
            else:
                self.view.floating = None
                self._finish_engine_move()

    def _finish_engine_move(self):
        nxt, pts, steps, eq, d1, d2 = self._engine_result
        played = format_steps(steps)
        human_eq = -eq  # eq is engine-perspective; show from your side
        if pts is not None:
            self._log_move("CPU", (d1, d2), steps, -float(pts))
            self._end_game(1, self.cube_value * pts, f"Engine wins {self._pts_name(pts)}")
            return
        self.board = nxt
        self.human_turn = True
        self.remaining = []
        self.busy = False
        self.turn_no += 1
        self._log_move("CPU", (d1, d2), steps, human_eq)
        self.refresh(f"Engine: {played}  (eq {human_eq:+.2f}). Click the dice to roll.")

    def _hint_rows(self):
        """Ranked moves for the current roll as [(notation, equity)], cached.

        Hovering shouldn't re-run the search: at 2-ply it's a fair fraction of a
        second, and the answer only changes when the position or roll does.
        """
        if not (self.human_turn and self.full_roll and self.remaining) or self.busy:
            return None
        key = (self.board.position_id(), self.roll)
        if self._hint_key != key:
            ranked = self.hint_engine.analyze(self.board, *self.roll)[:5]
            self._hint_key = key
            self._hint_cache = [(format_steps(s), float(eq)) for s, _, eq in ranked]
        return self._hint_cache

    def on_hint_hover(self, showing):
        self.view.hint_rows = self._hint_rows() if showing else None
        self.view.update()

    def on_hint(self):
        # Clicking does the same as hovering — the panel is the hint.
        self.on_hint_hover(True)

    def _log_move(self, who, roll, steps, eq_human):
        self.moves.addItem(f"{self.turn_no:2d}. {who} {roll[0]}-{roll[1]}  "
                           f"{format_steps(steps)}  [{eq_human:+.2f}]")
        self.moves.scrollToBottom()

    @staticmethod
    def _pts_name(pts):
        return {1: "a single", 2: "a gammon", 3: "a backgammon"}[pts]


def selftest(report_path: str) -> int:
    """Build the window offscreen, exercise the engine, and write a JSON report.

    A packaged windowed build has nowhere to print, and a missing `td.onnx`
    wouldn't crash it — the neural opponents would just quietly vanish. So the
    report names the engines that actually loaded and plays a move, and the
    release build is checked against it. Returns a process exit code.
    """
    import json
    import traceback

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    report = {"ok": False, "frozen": bool(getattr(sys, "frozen", False)), "root": str(ROOT)}
    try:
        QApplication(sys.argv[:1])
        win = MainWindow()
        report["opponents"] = list(win.opponents)
        report["evaluator"] = type(win.evaluator).__name__ if win.evaluator else None
        report["cube_rollouts"] = win._cube_ro is not None

        # Sound: "the object exists" proves nothing, so actually play one and
        # check the sink went active. (QSoundEffect used to claim Ready and
        # isPlaying here while emitting silence — hence the raw-PCM path.)
        report["sound"] = win.sfx.ok
        report["sound_volume"] = round(win.sfx.volume, 2)
        report["sound_device"] = win.sfx.device_name
        try:
            from PySide6.QtMultimedia import QMediaDevices

            report["audio_outputs"] = [d.description() for d in QMediaDevices.audioOutputs()]
        except Exception as e:
            report["audio_outputs"] = f"QtMultimedia failed: {e}"

        if win.sfx.ok:
            win.sfx.play_dice()
            # Look while it's still playing: the roll is only ~0.4s, so waiting
            # much longer catches it already finished and reads as silence.
            loop = QEventLoop()
            QTimer.singleShot(120, loop.quit)
            loop.exec()
            report["sound_plays"] = win.sfx.is_playing()
        # What the opponent selector actually shows on launch — the app should
        # come up playing its strongest engine.
        report["default_opponent"] = win.opp_box.currentText()
        report["hint_engine"] = win.hint_engine.name

        # Play a real move with the strongest neural engine that loaded.
        board = bgcore.Board.starting()
        eng = win.opponents.get("Neural — 2-ply") or win.hint_engine
        ranked = eng.analyze(board, 3, 1)
        report["engine"] = eng.name
        report["best_move_31"] = format_steps(ranked[0][0])
        report["equity"] = round(float(ranked[0][2]), 4)
        report["moves_ranked"] = len(ranked)
        report["torch_imported"] = "torch" in sys.modules
        report["ok"] = True
    except Exception:
        report["error"] = traceback.format_exc()

    Path(report_path).write_text(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--selftest":
        sys.exit(selftest(sys.argv[2]))
    app = QApplication(sys.argv)
    win = MainWindow()
    # Wide enough for the move list to show each move's equity without clipping.
    win.resize(1220, 720)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
