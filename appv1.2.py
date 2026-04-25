"""
appv1.2.py – Intelligent Deadlift Diagnosis System  v1.2
Design system: ui-ux-pro-max skill
  Style   : Dark Mode OLED  (WCAG AAA, ⚡ Excellent performance)
  Palette : Financial Dashboard – bg #020617, green #22C55E, amber #F59E0B
  Fonts   : Barlow Condensed (headings) / Barlow (body)
  Effects : minimal glow, dark-to-light transitions, 7:1+ text contrast

v1.2 adds: Nose Trajectory chart above Analysis panel, synced with video playback.
  - Y-axis: up = smaller pixel Y (person standing tall), down = larger pixel Y
  - Dashed amber lines mark detected rep segment boundaries
  - Solid light line tracks current video frame position
"""

import os
import sys
import uuid
import shutil
import tempfile

import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QSizePolicy, QSlider, QListWidget,
    QListWidgetItem, QMenu, QInputDialog, QMessageBox,
    QScrollArea, QFrame, QStackedWidget, QProgressBar, QFileDialog,
    QAbstractItemView, QGraphicsDropShadowEffect,
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize, QRect
from PyQt5.QtGui import (
    QPixmap, QImage, QColor, QPalette, QFont, QCursor,
    QDragEnterEvent, QDropEvent, QFontDatabase,
    QPainter, QPen,
)

from pipeline import DiagnosisEngine


ENGINE      = DiagnosisEngine()
STORAGE_DIR = os.path.join(tempfile.gettempdir(), "deadlift_history_v2")
os.makedirs(STORAGE_DIR, exist_ok=True)


# ── Palette ───────────────────────────────────────────────────────────────────
BG        = "#020617"
CARD      = "#0F172A"
SURFACE   = "#1E293B"
SURFACE2  = "#334155"
BORDER    = "#1E293B"
BORDER_A  = "#334155"

ACCENT    = "#22C55E"
ACC_DIM   = "#16A34A"
ACC_BG    = "#052E16"
ACC_GLOW  = "#4ADE80"

AMBER     = "#F59E0B"
AMBER_BG  = "#1C1000"

TEXT_P    = "#F8FAFC"
TEXT_S    = "#94A3B8"
TEXT_M    = "#475569"

DANGER    = "#EF4444"
INFO      = "#3B82F6"


# ── Font loading ───────────────────────────────────────────────────────────────
def _resolve_fonts():
    db = QFontDatabase()
    fams = db.families()
    title_font = "Barlow Condensed" if "Barlow Condensed" in fams else (
        "SF Pro Display" if "SF Pro Display" in fams else "Helvetica Neue"
    )
    body_font  = "Barlow" if "Barlow" in fams else (
        "SF Pro Text" if "SF Pro Text" in fams else "Helvetica Neue"
    )
    return title_font, body_font

_TITLE_FAM = "Helvetica Neue"
_BODY_FAM  = "Helvetica Neue"

def _make_fonts():
    global F_TITLE, F_HDR, F_BODY, F_SMALL, F_BADGE
    F_TITLE = QFont(_TITLE_FAM, 18, QFont.Bold)
    F_HDR   = QFont(_TITLE_FAM, 12, QFont.DemiBold)
    F_BODY  = QFont(_BODY_FAM,  12)
    F_SMALL = QFont(_BODY_FAM,  10)
    F_BADGE = QFont(_BODY_FAM,   9, QFont.Bold)

F_TITLE = F_HDR = F_BODY = F_SMALL = F_BADGE = QFont("Helvetica Neue", 12)


# ── Shared QSS ────────────────────────────────────────────────────────────────
def card_ss(r=12, border=None):
    b = border or BORDER
    return f"background:{CARD}; border:1px solid {b}; border-radius:{r}px;"

BTN_PRI = f"""
QPushButton {{
    background:{ACCENT}; color:#000; font-weight:bold;
    border:none; border-radius:8px; padding:7px 18px;
}}
QPushButton:hover   {{ background:{ACC_GLOW}; }}
QPushButton:pressed {{ background:{ACC_DIM};  }}
QPushButton:disabled{{ background:{BORDER};   color:{TEXT_M}; }}
QPushButton:focus   {{ outline:none; border:2px solid {ACC_GLOW}; }}
"""

BTN_SEC = f"""
QPushButton {{
    background:{SURFACE}; color:{TEXT_P};
    border:1px solid {BORDER_A}; border-radius:8px; padding:7px 16px;
}}
QPushButton:hover   {{ background:{SURFACE2}; border-color:{ACCENT}40; }}
QPushButton:pressed {{ background:{BORDER_A}; }}
QPushButton:focus   {{ border:1px solid {ACCENT}; outline:none; }}
"""

BTN_GHOST = f"""
QPushButton {{
    background:transparent; color:{TEXT_S};
    border:none; border-radius:6px; padding:4px 8px;
}}
QPushButton:hover   {{ background:{SURFACE}; color:{TEXT_P}; }}
QPushButton:pressed {{ background:{BORDER};  }}
"""

BTN_DANGER = f"""
QPushButton {{
    background:rgba(239,68,68,0.10); color:{DANGER};
    border:1px solid rgba(239,68,68,0.25); border-radius:8px; padding:7px 16px;
}}
QPushButton:hover {{ background:rgba(239,68,68,0.20); }}
"""

SLIDER_SS = f"""
QSlider::groove:horizontal {{
    height:4px; background:{SURFACE2}; border-radius:2px;
}}
QSlider::handle:horizontal {{
    width:14px; height:14px; margin:-5px 0;
    background:{ACCENT}; border-radius:7px;
}}
QSlider::handle:horizontal:hover {{ background:{ACC_GLOW}; }}
QSlider::sub-page:horizontal     {{ background:{ACCENT}; border-radius:2px; }}
"""

SCROLL_SS = f"""
QScrollArea {{ background:transparent; border:none; }}
QScrollBar:vertical {{
    background:{CARD}; width:5px; margin:0;
}}
QScrollBar::handle:vertical {{
    background:{SURFACE2}; border-radius:2px; min-height:20px;
}}
QScrollBar::handle:vertical:hover {{ background:{TEXT_M}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
"""

MENU_SS = f"""
QMenu {{
    background:{CARD}; color:{TEXT_P};
    border:1px solid {BORDER_A}; border-radius:10px; padding:4px;
}}
QMenu::item {{ padding:7px 20px; border-radius:6px; }}
QMenu::item:selected {{ background:{SURFACE}; color:{TEXT_P}; }}
QMenu::separator {{ height:1px; background:{BORDER_A}; margin:4px 10px; }}
"""


def _glow(widget, color=ACCENT, radius=18):
    fx = QGraphicsDropShadowEffect(widget)
    fx.setBlurRadius(radius)
    fx.setColor(QColor(color))
    fx.setOffset(0, 0)
    widget.setGraphicsEffect(fx)


# ═══════════════════════════════════════════════════════════════════════════════
# Data model
# ═══════════════════════════════════════════════════════════════════════════════
class HistoryEntry:
    def __init__(self, original_name, video_path):
        self.id            = str(uuid.uuid4())
        self.original_name = original_name
        self.display_name  = os.path.splitext(original_name)[0]
        self.video_path    = video_path
        self.reps          = []
        self.results       = []
        self.analyzed      = False
        self.nose_trace    = None  # (sample_frames: np.ndarray, nose_y: np.ndarray) | None


# ═══════════════════════════════════════════════════════════════════════════════
# Worker thread
# ═══════════════════════════════════════════════════════════════════════════════
class AnalysisWorker(QThread):
    progress = pyqtSignal(str, int)
    finished = pyqtSignal(list, list, object)   # reps, results, nose_trace
    error    = pyqtSignal(str)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path

    def run(self):
        try:
            reps, results = ENGINE.process_video(
                self.video_path,
                progress_cb=lambda msg, pct: self.progress.emit(msg, pct),
            )
            nose_trace = ENGINE._last_nose_trace
            self.finished.emit(reps, results, nose_trace)
        except Exception as exc:
            self.error.emit(str(exc))


# ═══════════════════════════════════════════════════════════════════════════════
# Upload area
# ═══════════════════════════════════════════════════════════════════════════════
class UploadArea(QFrame):
    files_dropped = pyqtSignal(list)

    _IDLE  = f"background:{CARD}; border:2px dashed {SURFACE2}; border-radius:16px;"
    _OVER  = f"background:{ACC_BG}; border:2px dashed {ACCENT}; border-radius:16px;"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(self._IDLE)
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignCenter)
        lay.setSpacing(10)

        icon = QLabel()
        icon_path = os.path.join(os.path.dirname(__file__), "materials", "upload_icon.png")
        if os.path.exists(icon_path):
            pix = QPixmap(icon_path).scaled(68, 68, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            icon.setPixmap(pix)
        else:
            icon.setText("⬆")
            icon.setFont(QFont("Helvetica Neue", 40))
            icon.setStyleSheet(f"color:{SURFACE2}; background:transparent; border:none;")
        icon.setAlignment(Qt.AlignCenter)

        title = QLabel("Drop video here")
        title.setFont(F_HDR)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            f"color:{TEXT_P}; background:transparent; border:none; letter-spacing:-0.2px;"
        )

        sub = QLabel("MP4 · MOV · AVI · MKV")
        sub.setFont(F_SMALL)
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet(f"color:{TEXT_M}; background:transparent; border:none;")

        btn = QPushButton("Browse files")
        btn.setFont(F_SMALL)
        btn.setStyleSheet(BTN_PRI)
        btn.setFixedWidth(128)
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(self._browse)

        lay.addStretch()
        lay.addWidget(icon, alignment=Qt.AlignCenter)
        lay.addSpacing(8)
        lay.addWidget(title)
        lay.addWidget(sub)
        lay.addSpacing(14)
        lay.addWidget(btn, alignment=Qt.AlignCenter)
        lay.addStretch()

    def _browse(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open Video", "",
            "Video Files (*.mp4 *.mov *.avi *.mkv *.MP4 *.MOV *.AVI *.MKV)"
        )
        if paths:
            self.files_dropped.emit(paths)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            self.setStyleSheet(self._OVER)

    def dragLeaveEvent(self, e):
        self.setStyleSheet(self._IDLE)

    def dropEvent(self, e):
        self.setStyleSheet(self._IDLE)
        valid = [
            u.toLocalFile() for u in e.mimeData().urls()
            if u.toLocalFile().lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
        ]
        if valid:
            self.files_dropped.emit(valid)
        e.acceptProposedAction()


# ═══════════════════════════════════════════════════════════════════════════════
# Video player
# ═══════════════════════════════════════════════════════════════════════════════
class VideoPlayer(QWidget):
    frame_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cap          = None
        self._total        = 0
        self._fps          = 30.0
        self._cur          = 0
        self._playing      = False
        self._user_seeking = False
        self._timer        = QTimer(self)
        self._timer.timeout.connect(self._next_frame)
        self._build()

    def _build(self):
        self.setStyleSheet(card_ss(12))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 10)
        lay.setSpacing(6)

        self.frame_lbl = QLabel()
        self.frame_lbl.setAlignment(Qt.AlignCenter)
        self.frame_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.frame_lbl.setStyleSheet("background:#000; border-radius:8px; border:none;")
        lay.addWidget(self.frame_lbl, stretch=10)

        self.time_lbl = QLabel("0:00 / 0:00")
        self.time_lbl.setFont(F_SMALL)
        self.time_lbl.setAlignment(Qt.AlignCenter)
        self.time_lbl.setStyleSheet(
            f"color:{TEXT_M}; background:transparent; border:none; font-variant-numeric:tabular-nums;"
        )
        lay.addWidget(self.time_lbl)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setStyleSheet(SLIDER_SS)
        self.slider.setCursor(Qt.PointingHandCursor)
        self.slider.sliderPressed.connect(lambda: setattr(self, "_user_seeking", True))
        self.slider.sliderReleased.connect(self._seek_released)
        self.slider.sliderMoved.connect(self._seek_moved)
        lay.addWidget(self.slider)

        ctrl = QHBoxLayout()
        ctrl.setSpacing(6)
        self.btn_play = self._mk_ctrl("▶")
        self.btn_stop = self._mk_ctrl("■")
        self.btn_play.clicked.connect(self._play_pause)
        self.btn_stop.clicked.connect(self._stop)
        ctrl.addWidget(self.btn_play)
        ctrl.addWidget(self.btn_stop)
        ctrl.addStretch()
        lay.addLayout(ctrl)

    def _mk_ctrl(self, text):
        b = QPushButton(text)
        b.setFont(QFont("Helvetica Neue", 13))
        b.setFixedSize(34, 34)
        b.setStyleSheet(BTN_SEC)
        b.setCursor(Qt.PointingHandCursor)
        return b

    def load(self, path):
        self._timer.stop()
        if self._cap:
            self._cap.release()
        self._cap     = cv2.VideoCapture(path)
        self._total   = max(1, int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self._fps     = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._cur     = 0
        self._playing = False
        self.btn_play.setText("▶")
        self.slider.setValue(0)
        self._show_frame(0)

    def _play_pause(self):
        if not self._cap:
            return
        if self._playing:
            self._timer.stop()
            self._playing = False
            self.btn_play.setText("▶")
        else:
            if self._cur >= self._total - 1:
                self._cur = 0
            self._timer.start(max(1, int(1000 / self._fps)))
            self._playing = True
            self.btn_play.setText("⏸")

    def _stop(self):
        self._timer.stop()
        self._playing = False
        self.btn_play.setText("▶")
        self._cur = 0
        self._show_frame(0)
        self.slider.setValue(0)

    def _next_frame(self):
        if not self._cap:
            return
        self._cur += 1
        if self._cur >= self._total:
            self._timer.stop()
            self._playing = False
            self.btn_play.setText("▶")
            self._cur = self._total - 1
            return
        self._show_frame(self._cur)
        if not self._user_seeking:
            self.slider.setValue(int(self._cur * 1000 / self._total))

    def _seek_released(self):
        self._user_seeking = False
        self._cur = max(0, min(int(self.slider.value() * self._total / 1000), self._total - 1))
        self._show_frame(self._cur)

    def _seek_moved(self, val):
        self._cur = max(0, min(int(val * self._total / 1000), self._total - 1))
        self._show_frame(self._cur)

    def _show_frame(self, fi):
        if not self._cap:
            return
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception:
            return
        h, w, ch = frame.shape
        img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)
        lw, lh = self.frame_lbl.width(), self.frame_lbl.height()
        if lw > 0 and lh > 0:
            pix = pix.scaled(lw, lh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.frame_lbl.setPixmap(pix)
        cur_s = fi / self._fps
        tot_s = self._total / self._fps
        self.time_lbl.setText(f"{_fmt(cur_s)} / {_fmt(tot_s)}")
        self.frame_changed.emit(fi)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._cap and not self._playing:
            self._show_frame(self._cur)


def _fmt(s):
    s = int(s)
    return f"{s // 60}:{s % 60:02d}"


# ═══════════════════════════════════════════════════════════════════════════════
# Rep result card
# ═══════════════════════════════════════════════════════════════════════════════
_FAULT_META = [
    ("Correct",        ACCENT,  ACC_BG),
    ("Hip first",      AMBER,   AMBER_BG),
    ("Knee dominant",  INFO,    "#060F2A"),
    ("Rounded back",   DANGER,  "#1A0505"),
]

class RepCard(QFrame):
    def __init__(self, rep_idx, result_vec, fps, rep_range, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        correct = bool(result_vec[0]) and not any(result_vec[1:])
        left_col = ACCENT if correct else DANGER
        self.setStyleSheet(
            f"background:{SURFACE}; "
            f"border:1px solid {left_col}30; "
            f"border-left:3px solid {left_col}; "
            f"border-radius:8px;"
        )

        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(8)

        num = QLabel(f"Rep {rep_idx}")
        num.setFont(F_BADGE)
        num.setFixedWidth(44)
        num.setStyleSheet(f"color:{TEXT_S}; background:transparent; border:none;")
        lay.addWidget(num)

        if rep_range and fps:
            dur = (rep_range[1] - rep_range[0]) / fps
            dur_lbl = QLabel(f"{dur:.1f}s")
            dur_lbl.setFont(F_SMALL)
            dur_lbl.setStyleSheet(
                f"color:{AMBER}; background:transparent; border:none; font-variant-numeric:tabular-nums;"
            )
            lay.addWidget(dur_lbl)

        lay.addStretch()

        for ci, (name, fg, bg) in enumerate(_FAULT_META):
            if result_vec[ci]:
                badge = QLabel(name)
                badge.setFont(F_BADGE)
                badge.setAlignment(Qt.AlignCenter)
                badge.setStyleSheet(
                    f"color:{fg}; background:{bg}; "
                    f"border:1px solid {fg}35; border-radius:5px; padding:2px 9px;"
                )
                lay.addWidget(badge)


# ═══════════════════════════════════════════════════════════════════════════════
# Nose Trajectory chart
# ═══════════════════════════════════════════════════════════════════════════════
class NoseChart(QWidget):
    """Custom painter widget that draws nose-Y over frame index."""

    _PL = 28   # left margin (Y-axis arrow labels)
    _PR = 8
    _PT = 4
    _PB = 4

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background:transparent; border:none;")
        self._sample_frames = np.array([], dtype=float)
        self._nose_y        = np.array([], dtype=float)
        self._rep_bounds    = []   # sorted list of frame indices (dashed lines)
        self._cur_frame     = 0
        self._total_frames  = 1

    def set_data(self, sample_frames, nose_y, reps, total_frames):
        if sample_frames is not None and nose_y is not None and len(sample_frames):
            self._sample_frames = np.asarray(sample_frames, dtype=float)
            self._nose_y        = np.asarray(nose_y,        dtype=float)
            bounds = set()
            for s, e in (reps or []):
                bounds.add(s)
                bounds.add(e)
            self._rep_bounds = sorted(bounds)
        else:
            self._sample_frames = np.array([], dtype=float)
            self._nose_y        = np.array([], dtype=float)
            self._rep_bounds    = []
        self._total_frames = max(1, int(total_frames))
        self._cur_frame    = 0
        self.update()

    def set_frame(self, fi):
        self._cur_frame = int(fi)
        self.update()

    def clear(self):
        self._sample_frames = np.array([], dtype=float)
        self._nose_y        = np.array([], dtype=float)
        self._rep_bounds    = []
        self._cur_frame     = 0
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        W, H = self.width(), self.height()
        PL, PR, PT, PB = self._PL, self._PR, self._PT, self._PB
        pw = W - PL - PR
        ph = H - PT - PB

        if pw <= 2 or ph <= 2:
            p.end()
            return

        # ── no data placeholder ───────────────────────────────────────────────
        if len(self._nose_y) == 0:
            p.setPen(QColor(TEXT_M))
            p.setFont(F_SMALL)
            p.drawText(QRect(PL, PT, pw, ph), Qt.AlignCenter, "Run analysis to view")
            p.end()
            return

        # ── coordinate helpers ────────────────────────────────────────────────
        y_lo = float(np.nanmin(self._nose_y))
        y_hi = float(np.nanmax(self._nose_y))
        if y_hi == y_lo:
            y_hi = y_lo + 1.0
        pad  = (y_hi - y_lo) * 0.08
        y_lo -= pad
        y_hi += pad
        x_max = float(self._total_frames)

        def _cx(fi):
            return PL + fi / x_max * pw

        def _cy(ny):
            # YOLO 0,0 = top-left → smaller pixel Y = person higher = chart top
            return PT + (ny - y_lo) / (y_hi - y_lo) * ph

        # ── subtle horizontal grid ────────────────────────────────────────────
        pen_grid = QPen(QColor(SURFACE2))
        pen_grid.setWidthF(0.5)
        p.setPen(pen_grid)
        for frac in (0.25, 0.5, 0.75):
            gy = int(PT + frac * ph)
            p.drawLine(PL, gy, PL + pw, gy)

        # ── rep boundary dashed lines (amber) ─────────────────────────────────
        pen_dash = QPen(QColor(AMBER))
        pen_dash.setStyle(Qt.DashLine)
        pen_dash.setWidthF(1.0)
        p.setPen(pen_dash)
        for bf in self._rep_bounds:
            x = int(_cx(bf))
            p.drawLine(x, PT, x, PT + ph)

        # ── nose Y curve (green) ──────────────────────────────────────────────
        pen_curve = QPen(QColor(ACCENT))
        pen_curve.setWidthF(1.5)
        pen_curve.setCapStyle(Qt.RoundCap)
        p.setPen(pen_curve)
        px_prev = py_prev = None
        for fi, ny in zip(self._sample_frames, self._nose_y):
            if np.isnan(ny):
                px_prev = py_prev = None
                continue
            cx_, cy_ = int(_cx(fi)), int(_cy(ny))
            if px_prev is not None:
                p.drawLine(px_prev, py_prev, cx_, cy_)
            px_prev, py_prev = cx_, cy_

        # ── current-frame playhead ────────────────────────────────────────────
        pen_ph = QPen(QColor(TEXT_S))
        pen_ph.setWidthF(1.0)
        p.setPen(pen_ph)
        phx = int(_cx(self._cur_frame))
        p.drawLine(phx, PT, phx, PT + ph)

        # ── Y-axis direction labels ───────────────────────────────────────────
        p.setPen(QColor(TEXT_M))
        p.setFont(F_SMALL)
        p.drawText(QRect(0, PT, PL - 3, 14), Qt.AlignRight | Qt.AlignVCenter, "↑")
        p.drawText(QRect(0, PT + ph - 14, PL - 3, 14), Qt.AlignRight | Qt.AlignVCenter, "↓")

        p.end()


class NoseTrajectoryPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        self.setStyleSheet(card_ss(12))
        self.setFixedHeight(128)
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 8)
        lay.setSpacing(4)

        hdr = QLabel("Movement Trajectory")
        hdr.setFont(F_HDR)
        hdr.setStyleSheet(f"color:{TEXT_P}; background:transparent; border:none;")
        lay.addWidget(hdr)

        self._chart = NoseChart()
        lay.addWidget(self._chart, stretch=1)

    def set_data(self, sample_frames, nose_y, reps, total_frames):
        self._chart.set_data(sample_frames, nose_y, reps, total_frames)

    def set_frame(self, fi):
        self._chart.set_frame(fi)

    def clear(self):
        self._chart.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis panel
# ═══════════════════════════════════════════════════════════════════════════════
class AnalysisPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        self.setStyleSheet(card_ss(12))
        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(8)

        hdr = QLabel("Analysis")
        hdr.setFont(F_HDR)
        hdr.setStyleSheet(f"color:{TEXT_P}; background:transparent; border:none;")
        outer.addWidget(hdr)

        self._summary = QLabel("—")
        self._summary.setFont(F_SMALL)
        self._summary.setStyleSheet(f"color:{TEXT_S}; background:transparent; border:none;")
        outer.addWidget(self._summary)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(SCROLL_SS)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._cards_w = QWidget()
        self._cards_w.setStyleSheet("background:transparent;")
        self._cards_l = QVBoxLayout(self._cards_w)
        self._cards_l.setContentsMargins(0, 0, 0, 0)
        self._cards_l.setSpacing(5)
        self._cards_l.addStretch()

        scroll.setWidget(self._cards_w)
        outer.addWidget(scroll, stretch=1)

    def show_results(self, results, reps=None, fps=30.0):
        self._clear_cards()
        if not results:
            self._summary.setText("No repetitions detected.")
            return
        n   = len(results)
        arr = np.array(results)
        ok  = int(((arr[:, 0] == 1) & (arr[:, 1:] == 0).all(axis=1)).sum())
        pct = int(ok * 100 / n)
        col = ACCENT if pct >= 70 else (AMBER if pct >= 40 else DANGER)
        self._summary.setText(
            f"<span style='color:{col}; font-weight:bold'>{ok}/{n}</span>"
            f"<span style='color:{TEXT_S}'> reps correct ({pct}%)</span>"
        )
        self._summary.setTextFormat(Qt.RichText)
        for i, rv in enumerate(results):
            rr = reps[i] if reps and i < len(reps) else None
            self._cards_l.insertWidget(self._cards_l.count() - 1, RepCard(i + 1, rv, fps, rr))

    def clear(self):
        self._clear_cards()
        self._summary.setText("—")

    def _clear_cards(self):
        while self._cards_l.count() > 1:
            item = self._cards_l.takeAt(0)
            if item.widget():
                item.widget().deleteLater()


# ═══════════════════════════════════════════════════════════════════════════════
# Feedback panel
# ═══════════════════════════════════════════════════════════════════════════════
_FB_ICONS = {
    "Hip":   ("▲", AMBER),
    "Knee":  ("◆", INFO),
    "Back":  ("!", DANGER),
    "Great": ("✓", ACCENT),
}

class FeedbackPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        self.setStyleSheet(card_ss(12))
        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(8)

        hdr = QLabel("Feedback")
        hdr.setFont(F_HDR)
        hdr.setStyleSheet(f"color:{TEXT_P}; background:transparent; border:none;")
        outer.addWidget(hdr)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(SCROLL_SS)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._body_w = QWidget()
        self._body_w.setStyleSheet("background:transparent;")
        self._body_l = QVBoxLayout(self._body_w)
        self._body_l.setContentsMargins(0, 0, 0, 0)
        self._body_l.setSpacing(5)

        self._ph = QLabel("—")
        self._ph.setFont(F_SMALL)
        self._ph.setStyleSheet(f"color:{TEXT_M}; background:transparent; border:none;")
        self._body_l.addWidget(self._ph)
        self._body_l.addStretch()

        scroll.setWidget(self._body_w)
        outer.addWidget(scroll, stretch=1)

    def show_feedback(self, text: str):
        self._clear()
        if not text or text.strip() == "—":
            self._ph.show()
            return
        self._ph.hide()
        for line in text.strip().split("\n"):
            line = line.strip()
            if line:
                self._body_l.insertWidget(self._body_l.count() - 1, self._row(line))

    def _row(self, text):
        ic, ic_col = "·", TEXT_M
        for key, (i, c) in _FB_ICONS.items():
            if key.lower() in text.lower():
                ic, ic_col = i, c
                break
        w = QWidget()
        w.setStyleSheet("background:transparent;")
        rl = QHBoxLayout(w)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(8)
        il = QLabel(ic)
        il.setFont(F_SMALL)
        il.setFixedWidth(14)
        il.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        il.setStyleSheet(f"color:{ic_col}; background:transparent; border:none;")
        tl = QLabel(text)
        tl.setFont(F_SMALL)
        tl.setWordWrap(True)
        tl.setStyleSheet(f"color:{TEXT_S}; background:transparent; border:none;")
        tl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        rl.addWidget(il, alignment=Qt.AlignTop)
        rl.addWidget(tl)
        return w

    def clear(self):
        self._clear()
        self._ph.show()

    def _clear(self):
        while self._body_l.count() > 0:
            item = self._body_l.takeAt(0)
            if item.widget() and item.widget() is not self._ph:
                item.widget().deleteLater()
        self._body_l.addWidget(self._ph)
        self._body_l.addStretch()


# ═══════════════════════════════════════════════════════════════════════════════
# History item widget
# ═══════════════════════════════════════════════════════════════════════════════
class HistoryItemWidget(QWidget):
    rename_requested  = pyqtSignal(str)
    delete_requested  = pyqtSignal(str)
    analyze_requested = pyqtSignal(str)

    def __init__(self, entry: HistoryEntry, parent=None):
        super().__init__(parent)
        self.entry_id  = entry.id
        self._analyzed = entry.analyzed
        self._build(entry.display_name)

    def _build(self, name):
        self.setStyleSheet("background:transparent;")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 4, 6, 4)
        lay.setSpacing(6)

        self._dot = QLabel("●")
        self._dot.setFixedWidth(10)
        self._dot.setFont(QFont("Helvetica Neue", 7))
        self._dot.setAlignment(Qt.AlignCenter)
        self._upd_dot()

        self.name_lbl = QLabel(name)
        self.name_lbl.setFont(F_SMALL)
        self.name_lbl.setStyleSheet(f"color:{TEXT_P}; background:transparent; border:none;")
        self.name_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        more = QPushButton("⋯")
        more.setFont(QFont("Helvetica Neue", 14))
        more.setFixedSize(26, 26)
        more.setStyleSheet(BTN_GHOST)
        more.setCursor(Qt.PointingHandCursor)
        more.clicked.connect(self._menu)

        lay.addWidget(self._dot)
        lay.addWidget(self.name_lbl)
        lay.addWidget(more)

    def _upd_dot(self):
        col = ACCENT if self._analyzed else TEXT_M
        self._dot.setStyleSheet(f"color:{col}; background:transparent; border:none;")

    def update_name(self, name, analyzed=None):
        self.name_lbl.setText(name)
        if analyzed is not None:
            self._analyzed = analyzed
            self._upd_dot()

    def _menu(self):
        m = QMenu(self)
        m.setStyleSheet(MENU_SS)
        if not self._analyzed:
            m.addAction("Analyze",  lambda: self.analyze_requested.emit(self.entry_id))
            m.addSeparator()
        m.addAction("Rename",  lambda: self.rename_requested.emit(self.entry_id))
        m.addAction("Delete",  lambda: self.delete_requested.emit(self.entry_id))
        m.exec_(QCursor.pos())


# ═══════════════════════════════════════════════════════════════════════════════
# History panel (sidebar)
# ═══════════════════════════════════════════════════════════════════════════════
class HistoryPanel(QFrame):
    entry_selected    = pyqtSignal(object)
    analyze_requested = pyqtSignal(object)
    upload_requested  = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        self.setFixedWidth(224)
        self.setStyleSheet(card_ss(12))
        self.setAcceptDrops(True)
        self._entries: dict[str, HistoryEntry] = {}
        self._widgets: dict[str, HistoryItemWidget] = {}
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 12, 0, 10)
        lay.setSpacing(6)

        hdr_row = QHBoxLayout()
        hdr_row.setContentsMargins(14, 0, 10, 0)
        hdr = QLabel("History")
        hdr.setFont(F_HDR)
        hdr.setStyleSheet(f"color:{TEXT_P}; background:transparent; border:none;")
        hdr_row.addWidget(hdr)
        hdr_row.addStretch()
        lay.addLayout(hdr_row)

        div = QFrame()
        div.setFrameShape(QFrame.HLine)
        div.setStyleSheet(f"color:{BORDER_A};")
        lay.addWidget(div)

        self.list_w = QListWidget()
        self.list_w.setDragDropMode(QAbstractItemView.InternalMove)
        self.list_w.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_w.setStyleSheet(
            f"QListWidget {{ background:transparent; border:none; outline:none; }}"
            f"QListWidget::item {{ border-radius:6px; }}"
            f"QListWidget::item:selected {{ background:{SURFACE}; }}"
            f"QListWidget::item:hover {{ background:{SURFACE}60; }}"
        )
        self.list_w.itemClicked.connect(self._clicked)
        self.list_w.itemDoubleClicked.connect(
            lambda item: self._rename(item.data(Qt.UserRole))
        )
        lay.addWidget(self.list_w, stretch=1)

        div2 = QFrame()
        div2.setFrameShape(QFrame.HLine)
        div2.setStyleSheet(f"color:{BORDER_A};")
        lay.addWidget(div2)

        btns = QWidget()
        btns.setStyleSheet("background:transparent;")
        bl = QVBoxLayout(btns)
        bl.setContentsMargins(10, 4, 10, 4)
        bl.setSpacing(5)

        self.btn_upload = QPushButton("+ Upload video")
        self.btn_upload.setFont(F_SMALL)
        self.btn_upload.setStyleSheet(BTN_PRI)
        self.btn_upload.setCursor(Qt.PointingHandCursor)
        self.btn_upload.clicked.connect(self._browse)

        row = QHBoxLayout()
        row.setSpacing(5)
        self.btn_all = QPushButton("Analyze all")
        self.btn_del = QPushButton("Delete all")
        for b in (self.btn_all, self.btn_del):
            b.setFont(F_SMALL)
            b.setCursor(Qt.PointingHandCursor)
        self.btn_all.setStyleSheet(BTN_SEC)
        self.btn_del.setStyleSheet(BTN_DANGER)
        self.btn_all.clicked.connect(self._analyze_all)
        self.btn_del.clicked.connect(self._delete_all)
        row.addWidget(self.btn_all)
        row.addWidget(self.btn_del)

        bl.addWidget(self.btn_upload)
        bl.addLayout(row)
        lay.addWidget(btns)

    # ── public API ──────────────────────────────────────────────────────────────
    def add_entry(self, entry: HistoryEntry):
        self._entries[entry.id] = entry
        item = QListWidgetItem()
        item.setData(Qt.UserRole, entry.id)
        item.setSizeHint(QSize(0, 38))
        self.list_w.insertItem(0, item)
        w = HistoryItemWidget(entry)
        w.rename_requested.connect(self._rename)
        w.delete_requested.connect(self._delete)
        w.analyze_requested.connect(self._analyze_one)
        self.list_w.setItemWidget(item, w)
        self._widgets[entry.id] = w

    def refresh_entry(self, entry: HistoryEntry):
        if entry.id in self._widgets:
            self._widgets[entry.id].update_name(entry.display_name, entry.analyzed)

    def get_unanalyzed(self):
        return [e for e in self._entries.values() if not e.analyzed]

    # ── slots ────────────────────────────────────────────────────────────────────
    def _clicked(self, item):
        eid = item.data(Qt.UserRole)
        if eid and eid in self._entries:
            self.entry_selected.emit(self._entries[eid])

    def _rename(self, eid):
        e = self._entries.get(eid)
        if not e:
            return
        new, ok = QInputDialog.getText(self, "Rename", "New name:", text=e.display_name)
        if ok and new.strip():
            e.display_name = new.strip()
            self.refresh_entry(e)

    def _delete(self, eid):
        e = self._entries.pop(eid, None)
        if not e:
            return
        self._widgets.pop(eid, None)
        for i in range(self.list_w.count()):
            item = self.list_w.item(i)
            if item and item.data(Qt.UserRole) == eid:
                self.list_w.takeItem(i)
                break
        try:
            if os.path.exists(e.video_path):
                os.remove(e.video_path)
        except Exception:
            pass

    def _analyze_one(self, eid):
        e = self._entries.get(eid)
        if e:
            self.analyze_requested.emit(e)

    def _analyze_all(self):
        for e in self.get_unanalyzed():
            self.analyze_requested.emit(e)

    def _delete_all(self):
        for eid in list(self._entries):
            self._delete(eid)

    def _browse(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select video(s)", "",
            "Video files (*.mp4 *.mov *.avi *.mkv)"
        )
        if paths:
            self.upload_requested.emit(paths)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls() and any(
            u.toLocalFile().lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
            for u in e.mimeData().urls()
        ):
            e.acceptProposedAction()
        else:
            e.ignore()

    def dropEvent(self, e):
        valid = [
            u.toLocalFile() for u in e.mimeData().urls()
            if u.toLocalFile().lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
        ]
        if valid:
            self.upload_requested.emit(valid)
        e.acceptProposedAction()


# ═══════════════════════════════════════════════════════════════════════════════
# Progress overlay
# ═══════════════════════════════════════════════════════════════════════════════
class ProgressOverlay(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            f"background:rgba(2,6,23,215); border:1px solid {BORDER_A}; border-radius:14px;"
        )
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignCenter)
        lay.setContentsMargins(28, 22, 28, 22)
        lay.setSpacing(10)

        self._msg = QLabel("Initialising…")
        self._msg.setFont(F_BODY)
        self._msg.setAlignment(Qt.AlignCenter)
        self._msg.setStyleSheet(f"color:{TEXT_P}; background:transparent; border:none;")
        lay.addWidget(self._msg)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setFixedWidth(300)
        self._bar.setFixedHeight(5)
        self._bar.setTextVisible(False)
        self._bar.setStyleSheet(
            f"QProgressBar {{ background:{SURFACE}; border:none; border-radius:2px; }}"
            f"QProgressBar::chunk {{ background:qlineargradient("
            f"x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 {ACC_DIM},stop:1 {ACC_GLOW}); border-radius:2px; }}"
        )
        lay.addWidget(self._bar, alignment=Qt.AlignCenter)

        self._sub = QLabel("")
        self._sub.setFont(F_SMALL)
        self._sub.setAlignment(Qt.AlignCenter)
        self._sub.setStyleSheet(f"color:{TEXT_M}; background:transparent; border:none;")
        lay.addWidget(self._sub)

        self._dot_timer = QTimer(self)
        self._dot_timer.timeout.connect(self._tick)
        self._dots = 0
        self.hide()

    def _tick(self):
        self._dots = (self._dots % 3) + 1
        self._sub.setText("·" * self._dots)

    def update(self, msg, pct):
        self._msg.setText(msg)
        self._bar.setValue(pct)
        if not self.isVisible():
            self.show()
            self._dot_timer.start(380)

    def hide_overlay(self):
        self._dot_timer.stop()
        self.hide()


# ═══════════════════════════════════════════════════════════════════════════════
# Status strip
# ═══════════════════════════════════════════════════════════════════════════════
class StatusStrip(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(26)
        self.setStyleSheet(f"background:{CARD}; border-top:1px solid {BORDER};")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 0, 16, 0)

        self._lbl = QLabel("Ready")
        self._lbl.setFont(F_SMALL)
        self._lbl.setStyleSheet(f"color:{TEXT_M}; background:transparent; border:none;")
        lay.addWidget(self._lbl)
        lay.addStretch()

        ver = QLabel("v1.2")
        ver.setFont(F_SMALL)
        ver.setStyleSheet(f"color:{TEXT_M}; background:transparent; border:none;")
        lay.addWidget(ver)

    def set(self, text, color=None):
        col = color or TEXT_M
        self._lbl.setStyleSheet(f"color:{col}; background:transparent; border:none;")
        self._lbl.setText(text)


# ═══════════════════════════════════════════════════════════════════════════════
# Main window
# ═══════════════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Intelligent Deadlift Diagnosis System")
        self.resize(1200, 740)
        self.setMinimumSize(880, 580)
        self.setStyleSheet(f"QMainWindow {{ background:{BG}; }}")

        self._current_entry: HistoryEntry | None   = None
        self._active_entry:  HistoryEntry | None   = None
        self._worker:        AnalysisWorker | None = None
        self._queue:         list[HistoryEntry]    = []
        self._queued_ids:    set[str]              = set()
        self._cur_fps:       float                 = 30.0

        self._build()

    # ── construction ───────────────────────────────────────────────────────────
    def _build(self):
        root = QWidget()
        self.setCentralWidget(root)
        vlay = QVBoxLayout(root)
        vlay.setContentsMargins(0, 0, 0, 0)
        vlay.setSpacing(0)

        # ── title bar ──────────────────────────────────────────────────────────
        tbar = QFrame()
        tbar.setFixedHeight(54)
        tbar.setStyleSheet(f"background:{CARD}; border-bottom:1px solid {BORDER};")
        tb_lay = QHBoxLayout(tbar)
        tb_lay.setContentsMargins(18, 0, 18, 0)
        tb_lay.setSpacing(0)

        for col in (DANGER, AMBER, ACCENT):
            d = QLabel("●")
            d.setFont(QFont("Helvetica Neue", 8))
            d.setFixedSize(13, 13)
            d.setAlignment(Qt.AlignCenter)
            d.setStyleSheet(f"color:{col}; background:transparent; border:none;")
            tb_lay.addWidget(d)
            tb_lay.addSpacing(5)
        tb_lay.addSpacing(12)

        title_lbl = QLabel("Intelligent Deadlift Diagnosis System")
        title_lbl.setFont(F_TITLE)
        title_lbl.setStyleSheet(f"color:{TEXT_P}; background:transparent; border:none;")
        _glow(title_lbl, ACCENT, radius=22)
        tb_lay.addWidget(title_lbl)
        tb_lay.addStretch()

        vlay.addWidget(tbar)

        # ── content ────────────────────────────────────────────────────────────
        self._content = QWidget()
        self._content.setStyleSheet(f"background:{BG};")
        c_lay = QHBoxLayout(self._content)
        c_lay.setContentsMargins(12, 12, 12, 12)
        c_lay.setSpacing(10)

        self._upload = UploadArea()
        self._upload.files_dropped.connect(self._on_files)
        self._player = VideoPlayer()
        self._player.hide()
        self._left = QStackedWidget()
        self._left.addWidget(self._upload)   # 0
        self._left.addWidget(self._player)   # 1
        c_lay.addWidget(self._left, stretch=5)

        # middle column: nose chart (top) + analysis + feedback
        mid = QWidget()
        mid.setStyleSheet("background:transparent;")
        ml = QVBoxLayout(mid)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.setSpacing(10)

        self._nose_chart = NoseTrajectoryPanel()
        self._analysis   = AnalysisPanel()
        self._feedback   = FeedbackPanel()

        ml.addWidget(self._nose_chart, stretch=0)
        ml.addWidget(self._analysis,   stretch=1)
        ml.addWidget(self._feedback,   stretch=1)

        self._mid = mid
        self._mid.hide()
        c_lay.addWidget(self._mid, stretch=4)

        self._hist = HistoryPanel()
        self._hist.entry_selected.connect(self._on_selected)
        self._hist.analyze_requested.connect(self._queue_analysis)
        self._hist.upload_requested.connect(self._on_files)
        self._hist.hide()
        c_lay.addWidget(self._hist, stretch=0)

        vlay.addWidget(self._content, stretch=1)

        # status bar
        self._status = StatusStrip()
        vlay.addWidget(self._status)

        # overlay
        self._overlay = ProgressOverlay(self._content)
        self._overlay.resize(390, 130)
        self._center_overlay()

        # sync video playhead → nose chart
        self._player.frame_changed.connect(self._nose_chart.set_frame)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._center_overlay()

    def _center_overlay(self):
        cw, ch = self._content.width(), self._content.height()
        ow, oh = self._overlay.width(), self._overlay.height()
        self._overlay.move(max(0, (cw - ow) // 2), max(0, (ch - oh) // 2))

    # ── file upload ────────────────────────────────────────────────────────────
    def _on_files(self, paths):
        if not paths:
            return
        last = None
        for p in paths:
            fname = os.path.basename(p)
            dst   = os.path.join(STORAGE_DIR, f"{uuid.uuid4()}_{fname}")
            shutil.copy2(p, dst)
            e = HistoryEntry(fname, dst)
            self._hist.add_entry(e)
            last = e
        self._hist.show()
        if last:
            self._current_entry = last
            self._left.setCurrentIndex(0)
            self._mid.hide()
            self._status.set(f"Added {len(paths)} video(s).  Click Analyze to start.", TEXT_S)

    # ── analysis queue ─────────────────────────────────────────────────────────
    def _queue_analysis(self, entry: HistoryEntry):
        if not entry or entry.analyzed:
            return
        if self._worker and self._worker.isRunning() and \
                self._current_entry and self._current_entry.id == entry.id:
            return
        if entry.id in self._queued_ids:
            return
        self._queue.append(entry)
        self._queued_ids.add(entry.id)
        self._next()

    def _next(self):
        if self._worker and self._worker.isRunning():
            return
        while self._queue:
            e = self._queue.pop(0)
            self._queued_ids.discard(e.id)
            if not e.analyzed:
                self._start(e)
                return

    def _start(self, entry: HistoryEntry):
        if self._worker and self._worker.isRunning():
            return
        self._active_entry  = entry
        self._current_entry = entry
        self._player.load(entry.video_path)
        self._left.setCurrentIndex(1)
        self._nose_chart.clear()
        self._analysis.clear()
        self._feedback.clear()
        self._mid.show()
        self._overlay.update("Initialising…", 0)
        self._center_overlay()
        self._status.set(f"Analysing  {entry.display_name}…", ACCENT)
        self._worker = AnalysisWorker(entry.video_path)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, msg, pct):
        self._overlay.update(msg, pct)
        self._center_overlay()

    def _on_done(self, reps, results, nose_trace):
        print(f"[_on_done] reps={len(reps)} results={len(results)} nose_trace={'set' if nose_trace is not None else 'None'}", flush=True)
        print(f"[_on_done] active_entry={'set' if self._active_entry else 'None'}  current_entry={'set' if self._current_entry else 'None'}", flush=True)
        self._overlay.hide_overlay()
        done = self._active_entry
        self._active_entry = None
        if done:
            done.reps       = reps
            done.results    = results
            done.nose_trace = nose_trace
            done.analyzed   = True
            self._cur_fps   = self._player._fps
            self._hist.refresh_entry(done)
            ids_match = self._current_entry and self._current_entry.id == done.id
            print(f"[_on_done] ids_match={ids_match}", flush=True)
            if ids_match:
                self._display(done)
            self._status.set(f"Done  —  {done.display_name}", ACCENT)
        self._next()

    def _on_error(self, msg):
        self._overlay.hide_overlay()
        self._active_entry = None
        self._status.set(f"Error: {msg}", DANGER)
        QMessageBox.critical(self, "Analysis Error", msg)
        self._next()

    # ── display ────────────────────────────────────────────────────────────────
    def _on_selected(self, entry: HistoryEntry):
        self._current_entry = entry
        self._player.load(entry.video_path)
        self._left.setCurrentIndex(1)
        self._nose_chart.clear()
        self._analysis.clear()
        self._feedback.clear()
        self._mid.show()
        if entry.analyzed:
            self._cur_fps = self._player._fps
            self._display(entry)

    def _display(self, entry: HistoryEntry):
        print(f"[_display] results={len(entry.results)} reps={len(entry.reps)} fps={self._cur_fps} total_frames={self._player._total}", flush=True)
        self._analysis.show_results(entry.results, entry.reps, self._cur_fps)
        print("[_display] show_results done", flush=True)
        self._feedback.show_feedback(DiagnosisEngine.generate_feedback(entry.results))
        print("[_display] show_feedback done", flush=True)
        if entry.nose_trace is not None:
            sf, ny = entry.nose_trace
            print(f"[_display] set_data: {len(sf)} samples", flush=True)
            self._nose_chart.set_data(sf, ny, entry.reps, self._player._total)
        else:
            print("[_display] nose_trace is None → clear()", flush=True)
            self._nose_chart.clear()
        print("[_display] done", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    global _TITLE_FAM, _BODY_FAM
    _TITLE_FAM, _BODY_FAM = _resolve_fonts()
    _make_fonts()

    pal = QPalette()
    pal.setColor(QPalette.Window,          QColor(BG))
    pal.setColor(QPalette.WindowText,      QColor(TEXT_P))
    pal.setColor(QPalette.Base,            QColor(CARD))
    pal.setColor(QPalette.AlternateBase,   QColor(SURFACE))
    pal.setColor(QPalette.Text,            QColor(TEXT_P))
    pal.setColor(QPalette.Button,          QColor(SURFACE))
    pal.setColor(QPalette.ButtonText,      QColor(TEXT_P))
    pal.setColor(QPalette.Highlight,       QColor(ACCENT))
    pal.setColor(QPalette.HighlightedText, QColor("#000"))
    pal.setColor(QPalette.ToolTipBase,     QColor(CARD))
    pal.setColor(QPalette.ToolTipText,     QColor(TEXT_P))
    pal.setColor(QPalette.Link,            QColor(ACCENT))
    app.setPalette(pal)

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
