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

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "trainer"))

from PySide6.QtCore import Qt, QPointF, QRectF, QTimer
from PySide6.QtGui import QAction, QBrush, QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import bgcore
from cube import should_double, should_take
from engine_api import HceEngine, NeuralEngine, RandomEngine, RolloutEngine, format_steps
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
DEST_FILL = QColor(70, 220, 120, 150)


class BoardView(QWidget):
    def __init__(self, on_click, on_dice):
        super().__init__()
        self.on_click, self.on_dice = on_click, on_dice
        self.setMinimumSize(760, 560)
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
        dice_rect = QRectF(W * 0.70 - (s + 6), H / 2 - s / 2, 2 * s + 12, s)
        return {
            "W": W, "H": H, "margin": margin, "pw": pw, "ph": ph, "r": r,
            "xL": xL, "bar": (bar_x0, bar_x1), "off": (off_x0, off_x0 + off_w),
            "dice_rect": dice_rect, "dice_s": s,
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
        self._draw_pips(p, g)
        if self.floating:
            x, y, human = self.floating
            self._disc(p, x, y, g["r"], human)
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
        r = g["r"] * 1.4
        x = g["margin"] + g["pw"] * 0.5
        if self.cube_owner == 1:      # engine owns -> top
            y = g["margin"] + g["ph"] + r + 6
        elif self.cube_owner == 0:    # you own -> bottom
            y = g["H"] - g["margin"] - g["ph"] - r - 6
        else:                          # centered
            y = g["H"] / 2
        p.setBrush(QBrush(QColor("#f5f0e1")))
        p.setPen(QPen(QColor("#2b2b2b"), 2))
        p.drawRoundedRect(QRectF(x - r, y - r, 2 * r, 2 * r), 6, 6)
        p.setPen(QPen(QColor("#222")))
        p.setFont(QFont("Arial", int(r * 0.85), QFont.Bold))
        p.drawText(QRectF(x - r, y - r, 2 * r, 2 * r), Qt.AlignCenter, str(self.cube_value))
        if self.wink_cube and self.wink_on:    # pulse ring when the engine doubles
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(SRC_RING, 4))
            p.drawRoundedRect(QRectF(x - r - 5, y - r - 5, 2 * r + 10, 2 * r + 10), 9, 9)

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
            win_x = x if d_you > d_eng else xe      # ring the higher die (starts)
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(SRC_RING, 3))
            p.drawRoundedRect(QRectF(win_x - 4, y - 4, s + 8, s + 8), 8, 8)
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

    def _draw_pips(self, p, g):
        p.setPen(QPen(QColor("#eafff2")))
        p.setFont(QFont("Arial", 11, QFont.Bold))
        p.drawText(QRectF(g["W"] * 0.10, g["H"] - g["margin"] - 22, 220, 20),
                   Qt.AlignLeft, f"You: {self.board.pip_count(0)} pips")
        p.drawText(QRectF(g["W"] * 0.10, g["margin"] + 2, 220, 20),
                   Qt.AlignLeft, f"Engine: {self.board.pip_count(1)} pips")

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

    def mousePressEvent(self, ev):
        x, y = ev.position().x(), ev.position().y()
        if ev.button() == Qt.RightButton:
            self.on_click(None, "right")
            return
        g = self._geom
        if g is not None and g["dice_rect"].contains(QPointF(x, y)):
            self.on_dice()
            return
        self.on_click(self.hit_test(x, y), "left")


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
        self.sfx = Sfx()

        file_menu = self.menuBar().addMenu("&File")
        new_act = QAction("&New Game", self)
        new_act.setShortcut("Ctrl+N")
        new_act.triggered.connect(self.new_game)
        quit_act = QAction("&Quit", self)
        quit_act.setShortcut("Ctrl+Q")
        quit_act.triggered.connect(self.close)
        file_menu.addAction(new_act)
        file_menu.addSeparator()
        file_menu.addAction(quit_act)

        ckpt = ROOT / "models" / "td_latest.pt"
        self.opponents = {}
        self.evaluator = None
        neural1 = None
        if ckpt.exists():
            neural0 = NeuralEngine(ckpt, lookahead=0)
            neural1 = NeuralEngine(ckpt, lookahead=1, share=neural0)
            neural2 = NeuralEngine(ckpt, lookahead=2, share=neural0)
            for e in (neural0, neural1, neural2):
                self.opponents[e.name] = e
            self.evaluator = neural0
        onnx_path = ROOT / "models" / "td.onnx"
        self._cube_ro = None
        if hasattr(bgcore, "Rollouts") and onnx_path.exists():
            ro = RolloutEngine(onnx_path, trials=180, truncate_plies=9, candidates=5)
            self.opponents[ro.name] = ro
            # A dedicated rollout evaluator for cube decisions (no candidate filter).
            self._cube_ro = bgcore.Rollouts(str(onnx_path), 150, 9, 0, 7)
        self.opponents["HCE (heuristic)"] = HceEngine()
        self.opponents["Random"] = RandomEngine()
        self.hint_engine = neural1 or self.opponents["HCE (heuristic)"]

        self.view = BoardView(self.on_click, self.on_dice)
        self.roll_btn = QPushButton("Roll")
        self.double_btn = QPushButton("Double")
        self.take_btn = QPushButton("Take")
        self.drop_btn = QPushButton("Drop")
        self.hint_btn = QPushButton("Hint")
        self.new_btn = QPushButton("New Game")
        self.opp_box = QComboBox()
        self.opp_box.addItems(list(self.opponents.keys()))
        if self.hint_engine in self.opponents.values():
            self.opp_box.setCurrentText(self.hint_engine.name)
        self.roll_btn.clicked.connect(self.on_dice)
        self.double_btn.clicked.connect(self.on_double)
        self.take_btn.clicked.connect(self.on_take)
        self.drop_btn.clicked.connect(self.on_drop)
        self.hint_btn.clicked.connect(self.on_hint)
        self.new_btn.clicked.connect(self.new_game)
        self.take_btn.setVisible(False)
        self.drop_btn.setVisible(False)

        self.score_label = QLabel("You 0 — 0 Engine")
        self.score_label.setStyleSheet("font-weight:bold; padding:0 8px;")

        controls = QHBoxLayout()
        for w in (self.roll_btn, self.double_btn, self.take_btn, self.drop_btn,
                  self.hint_btn, self.new_btn, QLabel("Opponent:"), self.opp_box):
            controls.addWidget(w)
        controls.addStretch(1)
        controls.addWidget(self.score_label)

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
        self.busy = False
        self.score = [0, 0]          # cumulative points [you, engine]
        self.pending_double = False  # engine offered a double, awaiting take/drop
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
        if self.remaining:
            self.view.dice = list(self.remaining)
        else:
            self.view.dice = list(self.roll)   # roll-time keeps the previous roll (it winks)
        self.view.carrying = self.carrying
        self.view.source_points = (
            {s[0] for s in self.subs} if (self.human_turn and self.remaining) else set())
        self.view.dest_points = (
            {s[1] for s in self.subs if s[0] == self.carrying} if self.carrying is not None else set())
        self.view.floating = None
        self.view.cube_value = self.cube_value
        self.view.cube_owner = self.cube_owner
        self.view.opening = self.opening
        self.view.open_dice = self._open if self.opening else None
        if message is not None:
            self.status.setText(message)
        ready = (self.human_turn and not self.remaining and not self.game_over
                 and not self.busy and not self.opening)
        self.roll_btn.setEnabled(ready and not self.pending_double)
        self.double_btn.setEnabled(ready and not self.pending_double and self.may_double(0))
        self.take_btn.setVisible(self.pending_double)
        self.drop_btn.setVisible(self.pending_double)
        self.hint_btn.setEnabled(
            self.human_turn and bool(self.remaining) and self.full_roll and not self.game_over)
        self.score_label.setText(f"You {self.score[0]} — {self.score[1]} Engine")
        self.eval_bar.set_value(self._win_prob())
        self.view.wink_dice = roll_time
        self.view.wink_cube = self.pending_double
        if self.view.wink_dice or self.view.wink_cube:
            if not self._wink_timer.isActive():
                self.view.wink_on = True
                self._wink_timer.start()
        elif self._wink_timer.isActive():
            self._wink_timer.stop()
            self.view.wink_on = True
        self.view.update()

    # --- game flow ---
    def new_game(self):
        self._roll_timer.stop()
        self._anim_timer.stop()
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
        self.moves.clear()
        self.turn_no = 1
        self.cube_value = 1
        self.cube_owner = None       # None = centered, 0 = you, 1 = engine
        self.pending_double = False
        self.take_btn.setVisible(False)
        self.drop_btn.setVisible(False)
        # Opening roll — one die each; the higher die starts.
        while True:
            d_you, d_eng = self.rng.randint(1, 6), self.rng.randint(1, 6)
            if d_you != d_eng:
                break
        self._open = (d_you, d_eng)
        self._open_winner = 0 if d_you > d_eng else 1
        self.opening = True
        self.view.opening = True
        self.view.open_dice = (d_you, d_eng)
        starter = "You" if self._open_winner == 0 else "Engine"
        self.refresh(f"Opening roll — You {d_you}, Engine {d_eng}. {starter} start "
                     f"(click to continue).")
        QTimer.singleShot(1500, self._open_finish)

    def _open_finish(self):
        if not self.opening:
            return
        self.opening = False
        self.view.opening = False
        self.view.open_dice = None
        if self._open_winner == 0:                 # you start — roll your turn
            self.human_turn = True
            self.remaining = []
            self.roll = ()
            self.refresh("You start — roll the dice.")
        else:                                      # engine starts
            self.human_turn = False
            self.remaining = []
            self.busy = True
            self.refresh("Engine starts…")
            QTimer.singleShot(450, self.engine_turn)

    def may_double(self, side):
        return (not self.game_over and self.cube_value < 64
                and self.cube_owner in (None, side))

    def on_dice(self):
        if self.opening:                # click past the opening roll
            self._open_finish()
            return
        if not (self.human_turn and not self.remaining and not self.game_over and not self.busy):
            return
        self.busy = True
        self._wink_timer.stop()         # stop the roll prompt once rolling starts
        self.view.wink_dice = False
        self.sfx.play_dice()
        self._roll_final = (self.rng.randint(1, 6), self.rng.randint(1, 6))
        self._roll_frames = 16          # longer tumble
        self._roll_timer.start(55)

    def _roll_frame(self):
        self._roll_frames -= 1
        if self._roll_frames > 0:
            self.view.dice = [self.rng.randint(1, 6), self.rng.randint(1, 6)]
            self.view.update()
            return
        self._roll_timer.stop()
        d1, d2 = self._roll_final
        self.roll = (d1, d2)
        self.remaining = [d1] * 4 if d1 == d2 else [d1, d2]
        self.full_roll = True
        self.carrying = None
        self.human_steps = []
        self.busy = False
        self.subs = bgcore.submoves(self.board, self.remaining)
        if not self.subs:
            self.refresh(f"Rolled {d1}-{d2}: no legal move (dance).")
            QTimer.singleShot(700, self.end_human_turn)
            return
        self.refresh(f"Rolled {d1}-{d2}. Click a checker, then its destination.")

    def on_click(self, pid, button):
        if self.opening:                # click past the opening roll
            self._open_finish()
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
        elif pid in sources:
            self.carrying = pid                        # switch to a different source
            self.refresh()
        else:
            self.carrying = None                       # clicked elsewhere -> deselect
            self.refresh()

    def apply_submove(self, sub):
        frm, to, die, result = sub
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
        if self.may_double(1):
            eq = self._cube_eval(self.board)  # engine equity (rollout-based when available)
            if should_double(eq, True):
                self.pending_double = True
                self.busy = True
                self.refresh(f"Engine doubles to {self.cube_value * 2}! Take or Drop?")
                return
        self.engine_play()

    def engine_play(self):
        if self.game_over:
            return
        self.busy = True
        d1, d2 = self.rng.randint(1, 6), self.rng.randint(1, 6)
        self.roll = (d1, d2)
        nxt, pts, steps, eq = self.opponent.choose(self.board, d1, d2)
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

    def on_hint(self):
        if not (self.human_turn and self.full_roll and self.remaining) or self.busy:
            return
        d1, d2 = self.roll
        ranked = self.hint_engine.analyze(self.board, d1, d2)[:3]
        self.view.dest_points = {t for _, t, _ in ranked[0][0] if t != OFF}
        self.view.update()
        parts = [f"{format_steps(s)} ({eq:+.2f})" for s, _, eq in ranked]
        self.status.setText("Hint: " + "   ".join(parts))

    def _log_move(self, who, roll, steps, eq_human):
        self.moves.addItem(f"{self.turn_no:2d}. {who} {roll[0]}-{roll[1]}  "
                           f"{format_steps(steps)}  [{eq_human:+.2f}]")
        self.moves.scrollToBottom()

    @staticmethod
    def _pts_name(pts):
        return {1: "a single", 2: "a gammon", 3: "a backgammon"}[pts]


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1080, 700)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
