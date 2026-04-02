"""
crop_dialog.py  –  ArchMorph Professional
==========================================
Kettős vágó párbeszédablak.

Használat:
  1. Húzással rajzolj téglalapot az egyik képen.
  2. A másik képen automatikusan megjelenik az arányos párja.
  3. Mindkét téglalapot mozgathatod / sarokpontoknál átméretezheted.
  4. "✂ Vágás" gomb: mindkét képet a saját téglalapjára vágja,
     majd mindkettőt azonos pixelméretűre méretezi.

Az oldalarány (aspect ratio) megosztott: mindkét téglalap mindig
ugyanolyan arányú – ha az egyiket átméretezed, a másik is alkalmazkodik.
"""
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import cv2

from PyQt6.QtCore  import Qt, QPointF, QRectF, pyqtSignal
from PyQt6.QtGui   import (QBrush, QColor, QImage, QPainter, QPen, QPixmap)
from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

try:
    from TRANSLATIONS import tr
except ImportError:
    def tr(text: str) -> str: return text   # type: ignore[misc]


# ── Vizuális konstansok ───────────────────────────────────────────────────────
_COL_RECT    = QColor(255, 200,   0, 220)
_COL_FILL    = QColor(255, 200,   0,  35)
_COL_HANDLE  = QColor(255, 255, 255, 230)
_COL_BORDER  = QColor(140, 140, 140, 200)
_COL_BG      = QColor( 26,  31,  36)
_COL_LABEL   = QColor(180, 180, 180)
_HANDLE_DRAW = 8     # handle rajzolt mérete (px)
_HANDLE_HIT  = 14    # handle kattintási sugara (px)
_MIN_PX      = 8     # minimális téglalap pixelméret


# ══════════════════════════════════════════════════════════════════════════════
#  CropCanvas – egy képes panel interaktív téglalappal
# ══════════════════════════════════════════════════════════════════════════════

class CropCanvas(QWidget):
    """Egyetlen kép megjelenítése interaktív vágó-téglalappal."""

    rect_changed = pyqtSignal()   # téglalap bármilyen változásakor

    def __init__(self, title: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._title   = title
        self._img:    Optional[np.ndarray] = None
        self._pixmap: Optional[QPixmap]    = None
        self._rect:   Optional[QRectF]     = None   # kép pixelkoordinátái
        self._aspect: Optional[float]      = None   # megosztott w/h arány

        # drag állapot
        self._state    = "idle"   # idle | drawing | moving | resizing
        self._corner   = -1       # 0=BF 1=JF 2=JH 3=BH (bal-fent stb.)
        self._d0_img:  Optional[QPointF] = None   # húzás kezdete kép-koordinátában
        self._d0_wgt:  Optional[QPointF] = None   # húzás kezdete widget-koordinátában
        self._rect0:   Optional[QRectF]  = None   # téglalap a húzás kezdetén

        self.setMinimumSize(300, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

    # ── Nyilvános API ─────────────────────────────────────────────────────────

    def set_image(self, img: np.ndarray) -> None:
        self._img = img
        rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qi   = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qi)
        self.update()

    def set_rect(self, rect: Optional[QRectF]) -> None:
        self._rect = rect
        self.update()

    def get_rect(self) -> Optional[QRectF]:
        return self._rect.normalized() if self._rect else None

    def set_aspect(self, aspect: Optional[float]) -> None:
        self._aspect = aspect

    def get_cropped(self) -> Optional[np.ndarray]:
        if self._img is None or self._rect is None:
            return None
        r  = self._rect.normalized()
        h, w = self._img.shape[:2]
        x1 = max(0, int(round(r.left())))
        y1 = max(0, int(round(r.top())))
        x2 = min(w, int(round(r.right())))
        y2 = min(h, int(round(r.bottom())))
        if x2 - x1 < _MIN_PX or y2 - y1 < _MIN_PX:
            return None
        return self._img[y1:y2, x1:x2].copy()

    # ── Koordináta-konverzió ──────────────────────────────────────────────────

    def _disp(self) -> Optional[QRectF]:
        """A képet megjelenítő widget-terület (letterboxed)."""
        if self._pixmap is None:
            return None
        pw, ph = self._pixmap.width(), self._pixmap.height()
        ww, wh = self.width(), self.height()
        s  = min(ww / pw, wh / ph)
        dw, dh = pw * s, ph * s
        return QRectF((ww - dw) / 2, (wh - dh) / 2, dw, dh)

    def _w2i(self, p: QPointF) -> QPointF:
        """Widget-koordináta → kép-koordináta."""
        d = self._disp()
        if d is None or self._img is None:
            return p
        h, w = self._img.shape[:2]
        return QPointF((p.x() - d.left()) / d.width()  * w,
                       (p.y() - d.top())  / d.height() * h)

    def _i2w(self, p: QPointF) -> QPointF:
        """Kép-koordináta → widget-koordináta."""
        d = self._disp()
        if d is None or self._img is None:
            return p
        h, w = self._img.shape[:2]
        return QPointF(d.left() + p.x() / w * d.width(),
                       d.top()  + p.y() / h * d.height())

    def _rect_w(self) -> Optional[QRectF]:
        """Jelenlegi téglalap widget-koordinátában."""
        if self._rect is None:
            return None
        return QRectF(self._i2w(self._rect.topLeft()),
                      self._i2w(self._rect.bottomRight())).normalized()

    def _corners_w(self):
        rw = self._rect_w()
        if rw is None:
            return []
        return [rw.topLeft(), rw.topRight(), rw.bottomRight(), rw.bottomLeft()]

    def _hit_corner(self, pos: QPointF) -> int:
        for i, c in enumerate(self._corners_w()):
            if (pos - c).manhattanLength() <= _HANDLE_HIT:
                return i
        return -1

    def _hit_inside(self, pos: QPointF) -> bool:
        rw = self._rect_w()
        return rw is not None and rw.contains(pos)

    # ── Rajzolás ─────────────────────────────────────────────────────────────

    def paintEvent(self, _) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), _COL_BG)

        d = self._disp()
        if d and self._pixmap:
            p.drawPixmap(d.toRect(), self._pixmap)
        elif self._pixmap is None:
            p.setPen(_COL_LABEL)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       tr("Nincs kép betöltve"))

        # Felirat
        if self._title:
            p.setPen(_COL_LABEL)
            p.drawText(6, 18, self._title)

        rw = self._rect_w()
        if rw:
            # Kitöltés
            p.fillRect(rw, _COL_FILL)
            # Keret
            p.setPen(QPen(_COL_RECT, 2, Qt.PenStyle.SolidLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(rw)
            # Sarokpontok (handle-ök)
            hs = _HANDLE_DRAW
            p.setBrush(QBrush(_COL_HANDLE))
            p.setPen(QPen(_COL_BORDER, 1))
            for c in self._corners_w():
                p.drawRect(QRectF(c.x() - hs / 2, c.y() - hs / 2, hs, hs))

        p.end()

    # ── Egér-kezelés ─────────────────────────────────────────────────────────

    @staticmethod
    def _epos(event) -> QPointF:
        return event.position() if hasattr(event, "position") else QPointF(event.pos())

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self._img is None:
            return
        pos  = self._epos(event)
        ipos = self._w2i(pos)
        c    = self._hit_corner(pos)

        if c >= 0:
            self._state   = "resizing"
            self._corner  = c
            self._d0_wgt  = pos
            self._rect0   = QRectF(self._rect) if self._rect else None
        elif self._hit_inside(pos):
            self._state   = "moving"
            self._d0_img  = ipos
            self._rect0   = QRectF(self._rect) if self._rect else None
        else:
            self._state   = "drawing"
            self._d0_img  = ipos
            self._rect    = QRectF(ipos, ipos)
        self.update()

    def mouseMoveEvent(self, event) -> None:
        pos  = self._epos(event)
        ipos = self._w2i(pos)

        # Kurzor frissítése
        c = self._hit_corner(pos)
        if c >= 0:
            cursors = [
                Qt.CursorShape.SizeFDiagCursor, Qt.CursorShape.SizeBDiagCursor,
                Qt.CursorShape.SizeFDiagCursor, Qt.CursorShape.SizeBDiagCursor,
            ]
            self.setCursor(cursors[c])
        elif self._hit_inside(pos):
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            self.setCursor(Qt.CursorShape.CrossCursor)

        # ── Rajzolás ─────────────────────────────────────────────────────────
        if self._state == "drawing" and self._d0_img is not None:
            dx = ipos.x() - self._d0_img.x()
            dy = ipos.y() - self._d0_img.y()
            if self._aspect and (abs(dx) + abs(dy)) > 2:
                # Oldalarány megtartása
                if abs(dx) >= abs(dy):
                    dy = (dx / self._aspect) * (1 if dy >= 0 else -1)
                else:
                    dx = (dy * self._aspect) * (1 if dx >= 0 else -1)
            self._rect = QRectF(self._d0_img,
                                QPointF(self._d0_img.x() + dx,
                                        self._d0_img.y() + dy))
            self.rect_changed.emit()
            self.update()

        # ── Mozgatás ─────────────────────────────────────────────────────────
        elif self._state == "moving" and self._d0_img is not None and self._rect0 is not None:
            dx = ipos.x() - self._d0_img.x()
            dy = ipos.y() - self._d0_img.y()
            self._rect = self._rect0.translated(dx, dy)
            self.rect_changed.emit()
            self.update()

        # ── Átméretezés ──────────────────────────────────────────────────────
        elif self._state == "resizing" and self._d0_wgt is not None and self._rect0 is not None:
            delta = pos - self._d0_wgt
            r0w   = QRectF(self._i2w(self._rect0.topLeft()),
                           self._i2w(self._rect0.bottomRight())).normalized()
            rw    = QRectF(r0w)
            c     = self._corner

            if   c == 0: rw.setTopLeft(    r0w.topLeft()     + delta)
            elif c == 1: rw.setTopRight(   r0w.topRight()    + delta)
            elif c == 2: rw.setBottomRight(r0w.bottomRight() + delta)
            elif c == 3: rw.setBottomLeft( r0w.bottomLeft()  + delta)
            rw = rw.normalized()

            # Oldalarány kényszer
            if self._aspect and rw.height() > 1:
                new_h = rw.width() / self._aspect
                if c in (0, 1):          # felső sarok → alját rögzítjük
                    rw.setTop(rw.bottom() - new_h)
                else:                    # alsó sarok → tetejét rögzítjük
                    rw.setBottom(rw.top() + new_h)

            # Widget → kép koordináta
            self._rect = QRectF(self._w2i(rw.topLeft()),
                                self._w2i(rw.bottomRight())).normalized()
            self.rect_changed.emit()
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        self._state = "idle"


# ══════════════════════════════════════════════════════════════════════════════
#  CropDialog – a párbeszédablak
# ══════════════════════════════════════════════════════════════════════════════

class CropDialog(QDialog):
    """
    Kettős vágó párbeszédablak.

    Húzással rajzolj téglalapot valamelyik képre.
    A másik képen automatikusan megjelenik az arányos párja.
    Mindkét téglalapot mozgathatod és átméretezheted (közös oldalarány).
    A "Vágás" gomb mindkét képet a saját téglalapjára vágja,
    majd azonos pixelméretűre méretezi őket.
    """

    def __init__(self, img_a: np.ndarray, img_b: np.ndarray,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._img_a   = img_a
        self._img_b   = img_b
        self._aspect: Optional[float]      = None
        self._result_a: Optional[np.ndarray] = None
        self._result_b: Optional[np.ndarray] = None
        self._syncing = False    # visszacsatolás megakadályozása

        self.setWindowTitle(tr("Képterület kijelölése és vágása"))
        self.setMinimumSize(920, 540)
        self._build_ui()

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(6)
        root.setContentsMargins(8, 8, 8, 8)

        # Súgó szöveg
        info = QLabel(tr(
            "Húzással rajzolj téglalapot valamelyik képre  →  "
            "a másikon automatikusan megjelenik az arányos párja.  "
            "Mozgatás: húzd a belsejét  |  Átméretezés: húzd a sarokpontokat."
        ))
        info.setWordWrap(True)
        info.setStyleSheet("color:#8aacbc; font-size:11px; padding:2px 4px;")
        root.addWidget(info)

        # Két canvas egymás mellett
        canvases = QHBoxLayout()
        canvases.setSpacing(6)

        self.canvas_a = CropCanvas(tr("Kép  A"), self)
        self.canvas_b = CropCanvas(tr("Kép  B"), self)
        self.canvas_a.set_image(self._img_a)
        self.canvas_b.set_image(self._img_b)
        self.canvas_a.rect_changed.connect(self._on_a_changed)
        self.canvas_b.rect_changed.connect(self._on_b_changed)

        canvases.addWidget(self.canvas_a)
        canvases.addWidget(self.canvas_b)
        root.addLayout(canvases, 1)

        # Gombsor
        btns = QHBoxLayout()
        btns.setSpacing(8)

        self._lbl_info = QLabel(tr("Rajzolj téglalapot valamelyik képre"))
        self._lbl_info.setStyleSheet("color:#7aadcc; font-size:11px;")
        btns.addWidget(self._lbl_info, 1)

        btn_clear = QPushButton(tr("↺  Törlés"))
        btn_clear.setToolTip(tr("Mindkét téglalap törlése"))
        btn_clear.clicked.connect(self._clear)
        btns.addWidget(btn_clear)

        self._btn_crop = QPushButton(tr("✂  Vágás és méretegyeztetés"))
        self._btn_crop.setEnabled(False)
        self._btn_crop.setToolTip(tr(
            "Mindkét képet a saját téglalapjára vágja,\n"
            "majd azonos pixelméretűre méretezi őket."
        ))
        self._btn_crop.setStyleSheet(
            "QPushButton{background:#2e7d32;color:#eee;font-weight:bold;"
            "padding:5px 16px;border-radius:4px;border:none;}"
            "QPushButton:hover{background:#388e3c;}"
            "QPushButton:disabled{background:#2a3030;color:#555;}"
        )
        self._btn_crop.clicked.connect(self._do_crop)
        btns.addWidget(self._btn_crop)

        btn_cancel = QPushButton(tr("Mégse"))
        btn_cancel.clicked.connect(self.reject)
        btns.addWidget(btn_cancel)

        root.addLayout(btns)

    # ── Szinkronizáció ───────────────────────────────────────────────────────

    def _on_a_changed(self) -> None:
        if self._syncing:
            return
        r = self.canvas_a.get_rect()
        if r is None or r.width() < _MIN_PX or r.height() < _MIN_PX:
            return

        # Frissítjük a megosztott oldalarányt
        self._aspect = r.width() / max(r.height(), 1.0)
        self.canvas_a.set_aspect(self._aspect)
        self.canvas_b.set_aspect(self._aspect)

        # Ha B-nek még nincs téglalapja: arányosan elhelyezzük
        if self.canvas_b.get_rect() is None:
            self._syncing = True
            self.canvas_b.set_rect(self._proportional_rect(
                r, self._img_a, self._img_b))
            self._syncing = False

        self._refresh_state()

    def _on_b_changed(self) -> None:
        if self._syncing:
            return
        r = self.canvas_b.get_rect()
        if r is None or r.width() < _MIN_PX or r.height() < _MIN_PX:
            return

        self._aspect = r.width() / max(r.height(), 1.0)
        self.canvas_a.set_aspect(self._aspect)
        self.canvas_b.set_aspect(self._aspect)

        if self.canvas_a.get_rect() is None:
            self._syncing = True
            self.canvas_a.set_rect(self._proportional_rect(
                r, self._img_b, self._img_a))
            self._syncing = False

        self._refresh_state()

    @staticmethod
    def _proportional_rect(r: QRectF,
                           src_img: np.ndarray,
                           dst_img: np.ndarray) -> QRectF:
        """Forrás téglalapot átarányosít a célimagere (relatív koordinátákkal)."""
        hs, ws = src_img.shape[:2]
        hd, wd = dst_img.shape[:2]

        # Relatív középpont
        cx = r.center().x() / ws * wd
        cy = r.center().y() / hs * hd

        # Relatív méret
        sw = r.width()  / ws * wd
        sh = r.height() / hs * hd

        return QRectF(cx - sw / 2, cy - sh / 2, sw, sh)

    def _clear(self) -> None:
        self.canvas_a.set_rect(None)
        self.canvas_b.set_rect(None)
        self._aspect = None
        self.canvas_a.set_aspect(None)
        self.canvas_b.set_aspect(None)
        self._btn_crop.setEnabled(False)
        self._lbl_info.setText(tr("Rajzolj téglalapot valamelyik képre"))

    def _refresh_state(self) -> None:
        ra = self.canvas_a.get_rect()
        rb = self.canvas_b.get_rect()
        ok = (ra is not None and ra.width() > _MIN_PX and ra.height() > _MIN_PX and
              rb is not None and rb.width() > _MIN_PX and rb.height() > _MIN_PX)
        self._btn_crop.setEnabled(ok)
        if ok:
            wa, ha = int(ra.width()), int(ra.height())
            wb, hb = int(rb.width()), int(rb.height())
            tw, th = max(wa, wb), max(ha, hb)
            ar = self._aspect or 1.0
            self._lbl_info.setText(
                f"A: {wa}×{ha} px   |   B: {wb}×{hb} px   |   "
                f"AR: {ar:.3f}   →   {tr('kimenet')}: {tw}×{th} px"
            )
        else:
            self._lbl_info.setText(tr("Rajzolj téglalapot valamelyik képre"))

    # ── Vágás ────────────────────────────────────────────────────────────────

    def _do_crop(self) -> None:
        crop_a = self.canvas_a.get_cropped()
        crop_b = self.canvas_b.get_cropped()
        if crop_a is None or crop_b is None:
            QMessageBox.warning(self, tr("Hiba"),
                                tr("Érvénytelen vágási terület – "
                                   "ellenőrizd, hogy mindkét téglalap a képen belül van."))
            return

        # Mindkettőt a nagyobb méretre skálázzuk (Lanczos → legjobb minőség)
        tw = max(crop_a.shape[1], crop_b.shape[1])
        th = max(crop_a.shape[0], crop_b.shape[0])
        self._result_a = cv2.resize(crop_a, (tw, th), interpolation=cv2.INTER_LANCZOS4)
        self._result_b = cv2.resize(crop_b, (tw, th), interpolation=cv2.INTER_LANCZOS4)
        self.accept()

    # ── Eredmény ─────────────────────────────────────────────────────────────

    def get_results(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Visszaadja a vágott és méretegyeztetett (img_a, img_b) párokat."""
        return self._result_a, self._result_b
