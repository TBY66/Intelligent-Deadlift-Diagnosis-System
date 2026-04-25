"""
app.py – Intelligent Deadlift Diagnosis System (macOS GUI, Version 1.0)

Layout (three states):
  Status 1 – initial:   centered upload area
  Status 2 – uploaded:  upload area (left) + history panel (right)
  Status 3 – analyzed:  video player (left) + analysis+feedback (centre) + history (right)
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
    QListWidgetItem, QMenu, QAction, QInputDialog, QMessageBox,
    QScrollArea, QFrame, QStackedWidget, QProgressBar, QFileDialog,
    QAbstractItemView,
)
from PyQt5.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QUrl, QMimeData, QPoint,
    QSize,
)
from PyQt5.QtGui import (
    QPixmap, QImage, QColor, QPalette, QFont, QDrag, QCursor,
    QDragEnterEvent, QDropEvent,
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget

from pipeline import DiagnosisEngine


# ═══════════════════════════════════════════════════════════════════════════════
# Shared engine (loaded once)
# ═══════════════════════════════════════════════════════════════════════════════
ENGINE = DiagnosisEngine()

STORAGE_DIR = os.path.join(tempfile.gettempdir(), "deadlift_history")
os.makedirs(STORAGE_DIR, exist_ok=True)

# ── Style constants ─────────────────────────────────────────────────────────
BG       = "#FFFFFF"
TITLE_BG = "#E8F5E9"
PANEL_BG = "#F9F9F9"
BORDER   = "#CCCCCC"
ACCENT   = "#2E7D32"
BTN_BG   = "#E0E0E0"
BTN_HOVER = "#C8C8C8"

FONT_TITLE = QFont("Helvetica Neue", 20, QFont.Bold)
FONT_BODY  = QFont("Helvetica Neue", 13)
FONT_SMALL = QFont("Helvetica Neue", 11)


# ═══════════════════════════════════════════════════════════════════════════════
# Data model
# ═══════════════════════════════════════════════════════════════════════════════
class HistoryEntry:
    def __init__(self, original_name, video_path):
        self.id            = str(uuid.uuid4())
        self.original_name = original_name
        self.display_name  = os.path.splitext(original_name)[0]
        self.video_path    = video_path   # copy in STORAGE_DIR
        self.reps          = []
        self.results       = []
        self.analyzed      = False


# ═══════════════════════════════════════════════════════════════════════════════
# Worker thread
# ═══════════════════════════════════════════════════════════════════════════════
class AnalysisWorker(QThread):
    progress  = pyqtSignal(str, int)        # message, percent
    finished  = pyqtSignal(list, list)      # reps, results
    error     = pyqtSignal(str)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path

    def run(self):
        try:
            reps, results = ENGINE.process_video(
                self.video_path,
                progress_cb=lambda msg, pct: self.progress.emit(msg, pct),
            )
            self.finished.emit(reps, results)
        except Exception as exc:
            self.error.emit(str(exc))


# ═══════════════════════════════════════════════════════════════════════════════
# Upload / drop area
# ═══════════════════════════════════════════════════════════════════════════════
class UploadArea(QFrame):
    files_dropped = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._setup_ui()

    def _setup_ui(self):
        self.setStyleSheet(
            f"background:{PANEL_BG}; border:2px dashed {BORDER}; border-radius:12px;"
        )
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignCenter)

        title = QLabel("Upload Video:")
        title.setFont(FONT_BODY)
        title.setAlignment(Qt.AlignCenter)

        sub = QLabel("(.mp4, .mov, .avi)  —  multiple files supported")
        sub.setFont(FONT_SMALL)
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet("color:#888888;")

        icon_lbl = QLabel()
        icon_path = os.path.join(os.path.dirname(__file__), "materials", "upload_icon.png")
        if os.path.exists(icon_path):
            pix = QPixmap(icon_path).scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            icon_lbl.setPixmap(pix)
        else:
            icon_lbl.setText("⬆")
            icon_lbl.setFont(QFont("Helvetica Neue", 48))
        icon_lbl.setAlignment(Qt.AlignCenter)

        btn = QPushButton("Browse File")
        btn.setFont(FONT_SMALL)
        btn.setStyleSheet(
            f"QPushButton{{background:{BTN_BG};border-radius:6px;padding:6px 18px;}}"
            f"QPushButton:hover{{background:{BTN_HOVER};}}"
        )
        btn.clicked.connect(self._open_dialog)

        lay.addStretch()
        lay.addWidget(title)
        lay.addWidget(sub)
        lay.addSpacing(12)
        lay.addWidget(icon_lbl, alignment=Qt.AlignCenter)
        lay.addSpacing(12)
        lay.addWidget(btn, alignment=Qt.AlignCenter)
        lay.addStretch()

    def _open_dialog(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open Video", "",
            "Video Files (*.mp4 *.mov *.avi *.mkv *.MP4 *.MOV *.AVI *.MKV)"
        )
        if paths:
            self.files_dropped.emit(paths)

    # drag-and-drop ──────────────────────────────────────────────────────────
    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            self.setStyleSheet(
                f"background:#E8F5E9; border:2px dashed {ACCENT}; border-radius:12px;"
            )

    def dragLeaveEvent(self, e):
        self.setStyleSheet(
            f"background:{PANEL_BG}; border:2px dashed {BORDER}; border-radius:12px;"
        )

    def dropEvent(self, e: QDropEvent):
        self.setStyleSheet(
            f"background:{PANEL_BG}; border:2px dashed {BORDER}; border-radius:12px;"
        )
        valid = [
            url.toLocalFile() for url in e.mimeData().urls()
            if url.toLocalFile().lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
        ]
        if valid:
            self.files_dropped.emit(valid)
        e.acceptProposedAction()


# ═══════════════════════════════════════════════════════════════════════════════
# Video player (OpenCV + QTimer – works on macOS without extra backends)
# ═══════════════════════════════════════════════════════════════════════════════
class VideoPlayer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._cap      = None
        self._total    = 0
        self._fps      = 30.0
        self._cur      = 0          # current frame index
        self._playing  = False
        self._user_seeking = False

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._next_frame)

        self._setup_ui()

    def _setup_ui(self):
        self.setStyleSheet(f"background:{PANEL_BG}; border:1px solid {BORDER}; border-radius:8px;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 8)

        # frame display
        self.frame_lbl = QLabel()
        self.frame_lbl.setAlignment(Qt.AlignCenter)
        self.frame_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.frame_lbl.setStyleSheet("background:black; border-radius:4px;")
        lay.addWidget(self.frame_lbl, stretch=10)

        # time label
        self.time_lbl = QLabel("0:00 / 0:00")
        self.time_lbl.setFont(FONT_SMALL)
        self.time_lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.time_lbl)

        # slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.sliderPressed.connect(self._slider_pressed)
        self.slider.sliderReleased.connect(self._slider_released)
        self.slider.sliderMoved.connect(self._slider_moved)
        self.slider.setStyleSheet(
            "QSlider::groove:horizontal{height:4px;background:#CCCCCC;border-radius:2px;}"
            "QSlider::handle:horizontal{width:14px;height:14px;background:#555;border-radius:7px;margin:-5px 0;}"
            "QSlider::sub-page:horizontal{background:#555;border-radius:2px;}"
        )
        lay.addWidget(self.slider)

        # controls
        ctrl = QHBoxLayout()
        ctrl.setSpacing(8)
        self.btn_play = self._mk_btn("▶", self._play_pause)
        self.btn_stop = self._mk_btn("■", self._stop)
        ctrl.addWidget(self.btn_play)
        ctrl.addWidget(self.btn_stop)
        ctrl.addStretch()
        lay.addLayout(ctrl)

    def _mk_btn(self, text, slot):
        b = QPushButton(text)
        b.setFont(QFont("Helvetica Neue", 14))
        b.setFixedSize(34, 34)
        b.setStyleSheet(
            f"QPushButton{{background:{BTN_BG};border-radius:6px;}}"
            f"QPushButton:hover{{background:{BTN_HOVER};}}"
        )
        b.clicked.connect(slot)
        return b

    # ── public ───────────────────────────────────────────────────────────────
    def load(self, path):
        self._timer.stop()
        if self._cap:
            self._cap.release()
        self._cap   = cv2.VideoCapture(path)
        self._total = max(1, int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self._fps   = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._cur   = 0
        self._playing = False
        self.btn_play.setText("▶")
        self.slider.setValue(0)
        self._show_frame(0)

    # ── playback ──────────────────────────────────────────────────────────────
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
            interval = max(1, int(1000 / self._fps))
            self._timer.start(interval)
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

    # ── seeking ───────────────────────────────────────────────────────────────
    def _slider_pressed(self):
        self._user_seeking = True

    def _slider_released(self):
        self._user_seeking = False
        frame = int(self.slider.value() * self._total / 1000)
        self._cur = max(0, min(frame, self._total - 1))
        self._show_frame(self._cur)

    def _slider_moved(self, val):
        frame = int(val * self._total / 1000)
        self._cur = max(0, min(frame, self._total - 1))
        self._show_frame(self._cur)

    # ── frame render ──────────────────────────────────────────────────────────
    def _show_frame(self, fi):
        if not self._cap:
            return
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = self._cap.read()
        if not ret:
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        img  = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
        pix  = QPixmap.fromImage(img)
        lw, lh = self.frame_lbl.width(), self.frame_lbl.height()
        if lw > 0 and lh > 0:
            pix = pix.scaled(lw, lh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.frame_lbl.setPixmap(pix)

        # update time label
        cur_s  = fi / self._fps
        tot_s  = self._total / self._fps
        self.time_lbl.setText(f"{self._fmt(cur_s)} / {self._fmt(tot_s)}")

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._cap and not self._playing:
            self._show_frame(self._cur)

    @staticmethod
    def _fmt(seconds):
        s = int(seconds)
        return f"{s // 60}:{s % 60:02d}"
        return f"{s // 60}:{s % 60:02d}"


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis panel
# ═══════════════════════════════════════════════════════════════════════════════
class AnalysisPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        self.setStyleSheet(f"background:{PANEL_BG}; border:1px solid {BORDER}; border-radius:8px;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 12)

        hdr = QLabel("Analysis")
        hdr.setFont(QFont("Helvetica Neue", 15, QFont.Bold))
        lay.addWidget(hdr)

        self.body = QLabel("—")
        self.body.setFont(FONT_BODY)
        self.body.setWordWrap(True)
        self.body.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        lay.addWidget(self.body, stretch=1)

    def show_results(self, results):
        if not results:
            self.body.setText("No repetitions detected.")
            return
        n     = len(results)
        arr   = np.array(results)
        names = ["Correct", "Hip first", "Knee dominant", "Rounded back"]
        lines = [f"Total reps: {n}"]
        for ci, name in enumerate(names):
            cnt = int(arr[:, ci].sum())
            if cnt > 0:
                lines.append(f"  {name}: {cnt}/{n}")
        self.body.setText("\n".join(lines))

    def clear(self):
        self.body.setText("—")


# ═══════════════════════════════════════════════════════════════════════════════
# Feedback panel
# ═══════════════════════════════════════════════════════════════════════════════
class FeedbackPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        self.setStyleSheet(f"background:{PANEL_BG}; border:1px solid {BORDER}; border-radius:8px;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 12)

        hdr = QLabel("Feedback")
        hdr.setFont(QFont("Helvetica Neue", 15, QFont.Bold))
        lay.addWidget(hdr)

        self.body = QLabel("—")
        self.body.setFont(FONT_BODY)
        self.body.setWordWrap(True)
        self.body.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        lay.addWidget(self.body, stretch=1)

    def show_feedback(self, text):
        self.body.setText(text)

    def clear(self):
        self.body.setText("—")


# ═══════════════════════════════════════════════════════════════════════════════
# History item widget
# ═══════════════════════════════════════════════════════════════════════════════
class HistoryItemWidget(QWidget):
    rename_requested = pyqtSignal(str)    # entry_id
    delete_requested = pyqtSignal(str)
    analyze_requested = pyqtSignal(str)

    def __init__(self, entry: HistoryEntry, parent=None):
        super().__init__(parent)
        self.entry_id = entry.id
        self._analyzed = entry.analyzed
        self._setup_ui(entry.display_name, entry.analyzed)

    def _setup_ui(self, name, analyzed):
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)

        self.name_lbl = QLabel(name)
        self.name_lbl.setFont(FONT_SMALL)
        self.name_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        if analyzed:
            self.name_lbl.setStyleSheet("color:#2E7D32;")
        else:
            self.name_lbl.setStyleSheet("color:#333333;")

        menu_btn = QPushButton("≡")
        menu_btn.setFont(QFont("Helvetica Neue", 14))
        menu_btn.setFixedSize(28, 28)
        menu_btn.setStyleSheet(
            f"QPushButton{{background:transparent;border:none;}}"
            f"QPushButton:hover{{background:{BTN_HOVER};border-radius:4px;}}"
        )
        menu_btn.clicked.connect(self._show_menu)

        lay.addWidget(self.name_lbl)
        lay.addWidget(menu_btn)

    def update_name(self, name, analyzed=None):
        self.name_lbl.setText(name)
        if analyzed is not None:
            self._analyzed = analyzed
            self.name_lbl.setStyleSheet(
                "color:#2E7D32;" if analyzed else "color:#333333;"
            )

    def _show_menu(self):
        menu = QMenu(self)
        if not self._analyzed:
            menu.addAction("Analyze", lambda: self.analyze_requested.emit(self.entry_id))
        menu.addAction("Rename",   lambda: self.rename_requested.emit(self.entry_id))
        menu.addAction("Delete",   lambda: self.delete_requested.emit(self.entry_id))
        menu.exec_(QCursor.pos())


# ═══════════════════════════════════════════════════════════════════════════════
# History panel
# ═══════════════════════════════════════════════════════════════════════════════
class HistoryPanel(QFrame):
    entry_selected    = pyqtSignal(object)   # HistoryEntry
    analyze_requested = pyqtSignal(object)   # HistoryEntry
    upload_requested  = pyqtSignal(list)     # list of file paths

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        self.setFixedWidth(200)
        self.setStyleSheet(f"background:{PANEL_BG}; border:1px solid {BORDER}; border-radius:8px;")
        self._entries: dict[str, HistoryEntry] = {}   # id → entry
        self._widgets: dict[str, HistoryItemWidget] = {}
        self.setAcceptDrops(True)
        self._setup_ui()

    def _setup_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 8, 6, 8)

        hdr = QLabel("History")
        hdr.setFont(QFont("Helvetica Neue", 13, QFont.Bold))
        lay.addWidget(hdr)

        self.list_widget = QListWidget()
        self.list_widget.setDragDropMode(QAbstractItemView.InternalMove)
        self.list_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_widget.setStyleSheet(
            "QListWidget{border:none;background:transparent;}"
            "QListWidget::item{border-radius:4px;}"
            "QListWidget::item:selected{background:#E8F5E9;}"
        )
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        self.list_widget.itemDoubleClicked.connect(self._on_double_click)
        lay.addWidget(self.list_widget, stretch=1)

        # bottom buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self.btn_analyze_all = QPushButton("Analyze all")
        self.btn_analyze_all.setFont(FONT_SMALL)
        self.btn_analyze_all.setStyleSheet(
            f"QPushButton{{background:{BTN_BG};border-radius:5px;padding:4px 8px;}}"
            f"QPushButton:hover{{background:{BTN_HOVER};}}"
        )
        self.btn_analyze_all.clicked.connect(self._analyze_all)

        self.btn_delete_all = QPushButton("Delete all")
        self.btn_delete_all.setFont(FONT_SMALL)
        self.btn_delete_all.setStyleSheet(
            "QPushButton{background:#FFCDD2;border-radius:5px;padding:4px 8px;}"
            "QPushButton:hover{background:#EF9A9A;}"
        )
        self.btn_delete_all.clicked.connect(self._delete_all)

        btn_row.addWidget(self.btn_analyze_all)
        btn_row.addWidget(self.btn_delete_all)
        lay.addLayout(btn_row)

        self.btn_upload = QPushButton("+ Upload video")
        self.btn_upload.setFont(FONT_SMALL)
        self.btn_upload.setStyleSheet(
            f"QPushButton{{background:{BTN_BG};border-radius:5px;padding:5px 8px;}}"
            f"QPushButton:hover{{background:{BTN_HOVER};}}"
        )
        self.btn_upload.clicked.connect(self._browse_upload)
        lay.addWidget(self.btn_upload)

    # ── public API ──────────────────────────────────────────────────────────
    def add_entry(self, entry: HistoryEntry):
        self._entries[entry.id] = entry

        item = QListWidgetItem()
        item.setData(Qt.UserRole, entry.id)
        item.setSizeHint(QSize(0, 36))
        self.list_widget.insertItem(0, item)

        w = HistoryItemWidget(entry)
        w.rename_requested.connect(self._rename)
        w.delete_requested.connect(self._delete)
        w.analyze_requested.connect(self._analyze_one)
        self.list_widget.setItemWidget(item, w)
        self._widgets[entry.id] = w

    def refresh_entry(self, entry: HistoryEntry):
        if entry.id in self._widgets:
            self._widgets[entry.id].update_name(entry.display_name, entry.analyzed)

    def get_all_entries(self):
        return list(self._entries.values())

    def get_unanalyzed(self):
        return [e for e in self._entries.values() if not e.analyzed]

    # ── slots ────────────────────────────────────────────────────────────────
    def _on_item_clicked(self, item):
        eid = item.data(Qt.UserRole)
        if eid and eid in self._entries:
            self.entry_selected.emit(self._entries[eid])

    def _on_double_click(self, item):
        eid = item.data(Qt.UserRole)
        if eid:
            self._rename(eid)

    def _rename(self, eid):
        entry = self._entries.get(eid)
        if not entry:
            return
        new_name, ok = QInputDialog.getText(
            self, "Rename", "New name:", text=entry.display_name
        )
        if ok and new_name.strip():
            entry.display_name = new_name.strip()
            self.refresh_entry(entry)

    def _delete(self, eid):
        entry = self._entries.pop(eid, None)
        if not entry:
            return
        self._widgets.pop(eid, None)
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item and item.data(Qt.UserRole) == eid:
                self.list_widget.takeItem(i)
                break
        try:
            if os.path.exists(entry.video_path):
                os.remove(entry.video_path)
        except Exception:
            pass

    # ── drag-and-drop ────────────────────────────────────────────────────────
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            if any(u.toLocalFile().lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
                   for u in e.mimeData().urls()):
                e.acceptProposedAction()
                return
        e.ignore()

    def dropEvent(self, e):
        valid = [
            url.toLocalFile() for url in e.mimeData().urls()
            if url.toLocalFile().lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
        ]
        if valid:
            self.upload_requested.emit(valid)
        e.acceptProposedAction()

    def _browse_upload(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select video(s)", "",
            "Video files (*.mp4 *.mov *.avi *.mkv)"
        )
        if paths:
            self.upload_requested.emit(paths)

    def _analyze_one(self, eid):
        entry = self._entries.get(eid)
        if entry:
            self.analyze_requested.emit(entry)

    def _analyze_all(self):
        for entry in self.get_unanalyzed():
            self.analyze_requested.emit(entry)

    def _delete_all(self):
        for eid in list(self._entries.keys()):
            self._delete(eid)


# ═══════════════════════════════════════════════════════════════════════════════
# Progress overlay
# ═══════════════════════════════════════════════════════════════════════════════
class ProgressOverlay(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            "background:rgba(255,255,255,220); border-radius:12px;"
        )
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignCenter)

        self.msg_lbl = QLabel("Initialising…")
        self.msg_lbl.setFont(FONT_BODY)
        self.msg_lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.msg_lbl)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setFixedWidth(280)
        self.bar.setStyleSheet(
            "QProgressBar{border:1px solid #CCCCCC;border-radius:4px;text-align:center;}"
            "QProgressBar::chunk{background:#2E7D32;border-radius:4px;}"
        )
        lay.addWidget(self.bar, alignment=Qt.AlignCenter)
        self.hide()

    def update(self, msg, pct):
        self.msg_lbl.setText(msg)
        self.bar.setValue(pct)
        self.show()

    def hide_overlay(self):
        self.hide()


# ═══════════════════════════════════════════════════════════════════════════════
# Main window
# ═══════════════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Intelligent Deadlift Diagnosis System")
        self.resize(1100, 680)
        self.setMinimumSize(800, 520)
        self.setStyleSheet(f"QMainWindow{{background:{BG};}}")

        self._current_entry: HistoryEntry | None = None
        self._active_analysis_entry: HistoryEntry | None = None
        self._worker: AnalysisWorker | None = None
        self._analysis_queue: list[HistoryEntry] = []
        self._queued_entry_ids: set[str] = set()

        self._build_ui()

    # ── UI construction ──────────────────────────────────────────────────────
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        root_lay = QVBoxLayout(root)
        root_lay.setContentsMargins(16, 12, 16, 12)
        root_lay.setSpacing(12)

        # ── title bar ────────────────────────────────────────────────────────
        title_frame = QFrame()
        title_frame.setStyleSheet(
            f"background:{TITLE_BG}; border-radius:16px;"
        )
        title_lay = QHBoxLayout(title_frame)
        title_lay.setContentsMargins(20, 10, 20, 10)
        title_lbl = QLabel("Intelligent Deadlift Diagnosis System")
        title_lbl.setFont(FONT_TITLE)
        title_lbl.setAlignment(Qt.AlignCenter)
        title_lay.addWidget(title_lbl)
        root_lay.addWidget(title_frame)

        # ── content area ─────────────────────────────────────────────────────
        self.content = QWidget()
        self.content_lay = QHBoxLayout(self.content)
        self.content_lay.setContentsMargins(0, 0, 0, 0)
        self.content_lay.setSpacing(12)
        root_lay.addWidget(self.content, stretch=1)

        # upload / video stack (left column)
        self.upload_area = UploadArea()
        self.upload_area.files_dropped.connect(self._on_files_dropped)

        self.video_player = VideoPlayer()
        self.video_player.hide()

        self.left_stack = QStackedWidget()
        self.left_stack.addWidget(self.upload_area)   # index 0
        self.left_stack.addWidget(self.video_player)  # index 1
        self.content_lay.addWidget(self.left_stack, stretch=5)

        # middle column (analysis + feedback)
        mid = QWidget()
        mid_lay = QVBoxLayout(mid)
        mid_lay.setContentsMargins(0, 0, 0, 0)
        mid_lay.setSpacing(10)
        self.analysis_panel = AnalysisPanel()
        self.feedback_panel = FeedbackPanel()
        mid_lay.addWidget(self.analysis_panel, stretch=1)
        mid_lay.addWidget(self.feedback_panel, stretch=1)
        self.mid_col = mid
        self.mid_col.hide()
        self.content_lay.addWidget(self.mid_col, stretch=4)

        # history panel (right column)
        self.history_panel = HistoryPanel()
        self.history_panel.entry_selected.connect(self._on_entry_selected)
        self.history_panel.analyze_requested.connect(self._queue_analysis)
        self.history_panel.upload_requested.connect(self._on_files_dropped)
        self.history_panel.hide()
        self.content_lay.addWidget(self.history_panel, stretch=0)

        # progress overlay (absolute positioned over content)
        self.overlay = ProgressOverlay(self.content)
        self.overlay.resize(360, 120)
        self._center_overlay()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._center_overlay()

    def _center_overlay(self):
        cw = self.content.width()
        ch = self.content.height()
        ow = self.overlay.width()
        oh = self.overlay.height()
        self.overlay.move(max(0, (cw - ow) // 2), max(0, (ch - oh) // 2))

    # ── file upload ──────────────────────────────────────────────────────────
    def _on_files_dropped(self, src_paths):
        if not src_paths:
            return

        last_entry = None
        for src_path in src_paths:
            # copy to storage
            fname = os.path.basename(src_path)
            dst   = os.path.join(STORAGE_DIR, f"{uuid.uuid4()}_{fname}")
            shutil.copy2(src_path, dst)

            entry = HistoryEntry(fname, dst)
            self.history_panel.add_entry(entry)
            last_entry = entry

        self.history_panel.show()

        if last_entry is None:
            return

        self._current_entry = last_entry

        # keep upload state after adding files; analysis is user-triggered
        self.left_stack.setCurrentIndex(0)
        self.mid_col.hide()

    # ── analysis ─────────────────────────────────────────────────────────────
    def _queue_analysis(self, entry: HistoryEntry):
        if not entry or entry.analyzed:
            return

        if self._current_entry and self._worker and self._worker.isRunning():
            if self._current_entry.id == entry.id:
                return

        if entry.id in self._queued_entry_ids:
            return

        self._analysis_queue.append(entry)
        self._queued_entry_ids.add(entry.id)
        self._start_next_analysis()

    def _start_next_analysis(self):
        if self._worker and self._worker.isRunning():
            return

        while self._analysis_queue:
            entry = self._analysis_queue.pop(0)
            self._queued_entry_ids.discard(entry.id)
            if entry.analyzed:
                continue
            self._start_analysis(entry)
            return

    def _start_analysis(self, entry: HistoryEntry):
        if self._worker and self._worker.isRunning():
            return

        self._active_analysis_entry = entry
        self._current_entry = entry
        self._set_ui_for_analysis(entry)
        self.overlay.update("Initialising…", 0)
        self._center_overlay()

        self._worker = AnalysisWorker(entry.video_path)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_analysis_done)
        self._worker.error.connect(self._on_analysis_error)
        self._worker.start()

    def _set_ui_for_analysis(self, entry: HistoryEntry):
        # show video player with the selected video
        self.video_player.load(entry.video_path)
        self.left_stack.setCurrentIndex(1)
        self.analysis_panel.clear()
        self.feedback_panel.clear()
        self.mid_col.show()

    def _on_progress(self, msg, pct):
        self.overlay.update(msg, pct)
        self._center_overlay()

    def _on_analysis_done(self, reps, results):
        self.overlay.hide_overlay()
        finished_entry = self._active_analysis_entry
        self._active_analysis_entry = None
        if finished_entry:
            finished_entry.reps     = reps
            finished_entry.results  = results
            finished_entry.analyzed = True
            self.history_panel.refresh_entry(finished_entry)
            if self._current_entry and self._current_entry.id == finished_entry.id:
                self._display_entry(finished_entry)
        self._start_next_analysis()

    def _on_analysis_error(self, msg):
        self.overlay.hide_overlay()
        self._active_analysis_entry = None
        QMessageBox.critical(self, "Analysis Error", msg)
        self._start_next_analysis()

    # ── entry display ─────────────────────────────────────────────────────────
    def _on_entry_selected(self, entry: HistoryEntry):
        self._current_entry = entry
        self._set_ui_for_analysis(entry)
        if entry.analyzed:
            self._display_entry(entry)

    def _display_entry(self, entry: HistoryEntry):
        self.analysis_panel.show_results(entry.results)
        feedback = DiagnosisEngine.generate_feedback(entry.results)
        self.feedback_panel.show_feedback(feedback)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # global palette
    pal = app.palette()
    pal.setColor(QPalette.Window, QColor(BG))
    pal.setColor(QPalette.WindowText, QColor("#1A1A1A"))
    app.setPalette(pal)

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
