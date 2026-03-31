"""
point_editor.py  –  Pontszerkesztő modul (ArchMorph Professional)
==================================================================
Önálló PyQt6 modul; csak numpy és opcionálisan cv2 kell hozzá.

Funkciók
--------
- Dupla bal klikk üres helyen → pontpár hozzáadása (A+B egyszerre, arányos tükrözéssel)
- Bal klikk + húzás ponton   → pont mozgatása
- Bal klikk + húzás képen   → gumiszalag-kijelölés (keret); érintett pontok kiemelve
- Ctrl + húzás képen         → ROI-kijelölés (helyi illesztéshez); sárga keret
- Jobb klikk ponton          → helyi menü (kiválasztás / törlés párjával)
- Ctrl+Görgetés              → pontindex léptetés (fel/le)
- Görgetés                   → zoom az aktív pont körül
- Delete                     → kijelölt pontpár(ok) törlése
- Ctrl+Z                     → visszavonás (legutóbbi add/delete)
- Nyílbillentyűk             → aktív pont pixelpontos mozgatása (Shift = 10 px)
- Space                      → zoom szinkron a másik canvasra
- Ctrl+0                     → zoom visszaállítás

Exportált osztályok
-------------------
    PointEditorCanvas   – egy képet megjelenítő szerkeszthető canvas
    PointEditorWidget   – kétpaneles szerkesztő (A + B kép, párkezelés)

Megjegyzés a layoutról
----------------------
A canvas QWidget-ből örökli magát (NEM QLabel-ből), és paintEvent()-et
használ a rajzoláshoz. Ez megakadályozza a QLabel.setPixmap() által
okozott végtelen layout-növekedést (pixmap → sizeHint loop).
"""
from __future__ import annotations

import math
from typing import List, Optional, Set, Tuple

import numpy as np

from PyQt6.QtCore import Qt, QRectF, QPoint, QSize, pyqtSignal
from PyQt6.QtGui import (
    QAction, QActionGroup, QBrush, QColor, QFont,
    QImage, QPainter, QPen, QPixmap, QPolygonF,
)
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QMenu, QPushButton,
    QSizePolicy, QSplitter, QVBoxLayout, QWidget,
)

# ── Vizuális konstansok ──────────────────────────────────────────────────────
_R_NORMAL = 6            # Normál pont sugara (px)
_R_ACTIVE = 10           # Aktív pont sugara (px)
_R_MULTI  = 8            # Multi-kijelölt pont sugara (px)
_HIT_R    = 14           # Kattintási érzékenység (px)

_C_NORMAL = QColor("#ff8a3d")           # narancs  – normál pont
_C_ACTIVE = QColor("#ff3333")           # piros    – aktív pont
_C_MULTI  = QColor("#44aaff")           # kék      – multi-kijelölt pont
_C_FILL_N = QColor(255, 138,  61,  70)  # narancs fill (áttetsző)
_C_FILL_A = QColor(255,  51,  51, 150)  # piros fill (aktív)
_C_FILL_M = QColor( 68, 170, 255,  90)  # kék fill (multi-kijelölt)
_C_TEXT   = QColor("#ffffff")
_C_TEXT_A = QColor("#ffcccc")
_C_BG     = QColor("#1a1f24")
_C_BORDER = QColor("#343b45")
_C_PLACEHOLDER = QColor("#4a5260")

# Gumiszalag stílus
_C_RBAND_SEL = QColor("#88bbff")        # kék keret – normál kijelölés
_C_RBAND_ROI = QColor("#ffcc44")        # sárga keret – Ctrl-ROI

# Sokszög-ROI stílus
_C_POLY_DRAW = QColor("#FFB300")        # borostyán – rajzolás alatt
_C_POLY_DONE = QColor("#FF6F00")        # mélyebb narancs – lezárt, menü aktív
_C_POLY_FILL = QColor(255, 179, 0, 35)  # áttetsző kitöltés


# ── Segédfüggvény (cv2 nélkül is működik) ───────────────────────────────────
def _bgr_to_qpixmap(image_bgr: np.ndarray) -> QPixmap:
    try:
        import cv2
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        rgb = image_bgr[:, :, ::-1].copy()
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


# ────────────────────────────────────────────────────────────────────────────
#  PointEditorCanvas  –  QWidget alapú, paintEvent() rajzol
# ────────────────────────────────────────────────────────────────────────────
class PointEditorCanvas(QWidget):
    """
    Egy képet megjelenítő canvas teljes pontszerkesztési funkcióval.

    QWidget + paintEvent() alapú — NEM QLabel — ezért a layout mérete
    mindig rögzített és nem igazodik a betöltött kép pixelméretéhez.

    Jelek
    -----
    point_added(x, y)               Új pont kérése (képkoordinátában)
    point_moved(idx, x, y)          Pont pozíciója megváltozott
    point_deleted(idx)              Pont törlése kérve (egyszeres)
    points_delete_multi_requested() Multi-kijelölt pontok törlése kérve
    point_selected(idx)             Pont kiválasztva (-1 = deselect)
    selection_changed(list[int])    Gumiszalag kijelölés végeredménye
    roi_search_requested(x1,y1,x2,y2)  Ctrl-bekeretezés (képkoord, ROI search)
    zoom_sync_requested(zoom, rx, ry)   Space: zoom-szinkron kérés
    middle_clicked()                Középső gomb: zoom reset + undo
    undo_requested()                Ctrl+Z: visszavonás
    """

    point_added                 = pyqtSignal(float, float)
    point_moved                 = pyqtSignal(int, float, float)
    point_deleted               = pyqtSignal(int)
    points_delete_multi_requested = pyqtSignal()
    point_selected              = pyqtSignal(int)
    selection_changed           = pyqtSignal(list)
    roi_ready                   = pyqtSignal(list)  # sokszög widget-koordinátákban
    zoom_sync_requested         = pyqtSignal(float, float, float)
    middle_clicked              = pyqtSignal()
    undo_requested              = pyqtSignal()

    _ZOOM_MIN  = 0.5
    _ZOOM_MAX  = 20.0
    _ZOOM_STEP = 1.18    # egy kerékgörgés szorzója
    _RBAND_MIN = 4       # minimális mozgás px-ben a gumiszalag aktiválásához

    def __init__(self, title: str) -> None:
        super().__init__()
        self.title = title

        self._pixmap_orig: Optional[QPixmap]           = None
        self._img_size:    Optional[Tuple[int, int]]   = None   # (w, h)
        self._points:      List[Tuple[float, float]]   = []
        self._selected:    int                         = -1
        self._selected_set: Set[int]                   = set()  # multi-kijelölés

        # Pont-húzás
        self._drag_idx: int  = -1
        self._dragging: bool = False

        # Zoom / pan
        self._zoom:   float = 1.0
        self._pan_cx: float = 0.0
        self._pan_cy: float = 0.0

        # Jobb gombos pásztázás
        self._rpan_active: bool  = False
        self._rpan_moved:  bool  = False
        self._rpan_lx:     float = 0.0
        self._rpan_ly:     float = 0.0

        # Gumiszalag-kijelölés
        self._rband_p0:     Optional[Tuple[float, float]] = None  # press pozíció
        self._rband_p1:     Optional[Tuple[float, float]] = None  # aktuális végpont
        self._rband_active: bool = False   # True ha elég mozgás volt
        self._rband_ctrl:   bool = False   # Ctrl nyomva a bekeretezés alatt

        # Sokszög-ROI rajzolás
        self._poly_pts:    List[Tuple[float, float]] = []   # csúcsok widget-koordinátában
        self._poly_mode:   bool                      = False # rajzolás aktív
        self._cursor_pos:  Tuple[float, float]       = (0.0, 0.0)  # preview vonal
        self._display_poly: List[Tuple[float, float]]= []   # lezárt, menü alatt mutatva

        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

    def sizeHint(self) -> QSize:
        return QSize(480, 360)

    # ── Publikus API ─────────────────────────────────────────────────────────

    def set_image(self, image_bgr: np.ndarray) -> None:
        self._pixmap_orig = _bgr_to_qpixmap(image_bgr)
        h, w = image_bgr.shape[:2]
        self._img_size = (w, h)
        self._zoom   = 1.0
        self._pan_cx = w / 2.0
        self._pan_cy = h / 2.0
        self.update()

    def set_points(self, points: List[Tuple[float, float]]) -> None:
        self._points = list(points)
        self.update()

    def set_selected(self, index: int) -> None:
        if self._selected != index:
            self._selected = index
            self.update()

    def set_selected_set(self, indices: List[int]) -> None:
        """Multi-kijelölés szinkronizálása a párcanvasról."""
        self._selected_set = set(indices)
        self.update()

    # ── Koordináta-transzformáció ────────────────────────────────────────────

    def _display_rect(self) -> Optional[QRectF]:
        if self._pixmap_orig is None or self._img_size is None:
            return None
        iw, ih = self._img_size
        ww, wh = self.width(), self.height()
        if min(iw, ih, ww, wh) <= 0:
            return None
        base_scale = min(ww / iw, wh / ih)
        eff_scale  = base_scale * self._zoom
        dw, dh     = iw * eff_scale, ih * eff_scale
        left       = ww / 2 - self._pan_cx * eff_scale
        top        = wh / 2 - self._pan_cy * eff_scale
        return QRectF(left, top, dw, dh)

    def _w2i(self, wx: float, wy: float) -> Optional[Tuple[float, float]]:
        """Widget-koordináta → képkoordináta (None ha kívül esik a képen)."""
        rect = self._display_rect()
        if rect is None or self._img_size is None:
            return None
        if not rect.contains(wx, wy):
            return None
        iw, ih = self._img_size
        rx = (wx - rect.left()) / rect.width()
        ry = (wy - rect.top())  / rect.height()
        return rx * iw, ry * ih

    def _i2w(self, ix: float, iy: float) -> Optional[Tuple[float, float]]:
        """Képkoordináta → widget-koordináta."""
        rect = self._display_rect()
        if rect is None or self._img_size is None:
            return None
        iw, ih = self._img_size
        return (
            rect.left() + (ix / iw) * rect.width(),
            rect.top()  + (iy / ih) * rect.height(),
        )

    def _hit_test(self, wx: float, wy: float) -> int:
        best_idx, best_dist = -1, float(_HIT_R)
        for i, (ix, iy) in enumerate(self._points):
            wpt = self._i2w(ix, iy)
            if wpt is None:
                continue
            d = math.hypot(wx - wpt[0], wy - wpt[1])
            if d < best_dist:
                best_dist, best_idx = d, i
        return best_idx

    # ── Egéresemények ────────────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        wx  = float(event.position().x())
        wy  = float(event.position().y())
        btn = event.button()

        # ── Középső gomb ─────────────────────────────────────────────────────
        if btn == Qt.MouseButton.MiddleButton:
            self._reset_zoom()
            self.middle_clicked.emit()
            return

        # ── Jobb gomb ────────────────────────────────────────────────────────
        if btn == Qt.MouseButton.RightButton:
            idx = self._hit_test(wx, wy)
            if idx >= 0:
                self._show_context_menu(idx, event.globalPosition().toPoint())
            else:
                self._rpan_active = True
                self._rpan_moved  = False
                self._rpan_lx     = wx
                self._rpan_ly     = wy
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        # ── Bal gomb ─────────────────────────────────────────────────────────
        if btn == Qt.MouseButton.LeftButton:
            idx = self._hit_test(wx, wy)
            if idx >= 0:
                # Kattintás meglévő ponton → húzás előkészítése
                self._drag_idx = idx
                self._dragging = False
                # Ha a pont nincs a multi-szetben, töröljük a multi-szetet
                if idx not in self._selected_set:
                    self._selected_set.clear()
                self.point_selected.emit(idx)
            else:
                # Kattintás üres területen → gumiszalag előkészítése
                # (aktiválódik mouseMoveEvent-ben, ha elég mozgás van)
                coords = self._w2i(wx, wy)
                if coords is not None:
                    self._rband_p0    = (wx, wy)
                    self._rband_p1    = (wx, wy)
                    self._rband_ctrl  = bool(
                        event.modifiers() & Qt.KeyboardModifier.ControlModifier)
                    self._rband_active = False

        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        """
        Ctrl+dupla-klikk sokszög módban → sokszög lezárása + ROI menü.
        Dupla kattintás üres területen (Ctrl nélkül) → új pontpár hozzáadása.
        """
        if event.button() == Qt.MouseButton.LeftButton:
            wx = float(event.position().x())
            wy = float(event.position().y())
            ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)

            # Ctrl+dupla-klikk sokszög módban → lezárás
            if ctrl and self._poly_mode and len(self._poly_pts) >= 3:
                self._close_polygon()
                self._rband_active = False
                self._rband_p0     = None
                self._rband_p1     = None
                super().mouseDoubleClickEvent(event)
                return

            # Ha sokszög mód aktív de Ctrl nélküli dupla-klikk → megszakítás
            if self._poly_mode:
                self._poly_mode = False
                self._poly_pts  = []
                self._rband_active = False
                self._rband_p0     = None
                self._rband_p1     = None
                self.update()
                super().mouseDoubleClickEvent(event)
                return

            # Normál dupla-klikk: gumiszalag törlése, pont hozzáadása
            self._rband_active = False
            self._rband_p0     = None
            self._rband_p1     = None
            idx = self._hit_test(wx, wy)
            if idx < 0:
                coords = self._w2i(wx, wy)
                if coords is not None:
                    self.point_selected.emit(-1)
                    self.point_added.emit(*coords)
        super().mouseDoubleClickEvent(event)

    def _close_polygon(self) -> None:
        """Sokszög lezárása és roi_ready jel küldése."""
        if len(self._poly_pts) < 3:
            return
        poly = list(self._poly_pts)
        self._poly_mode    = False
        self._poly_pts     = []
        self._display_poly = poly   # megőrzi amíg a menü megjelenik
        self.update()
        self.roi_ready.emit(poly)

    def mouseMoveEvent(self, event) -> None:
        wx = float(event.position().x())
        wy = float(event.position().y())

        # Kurzorpozíció frissítése (sokszög preview vonalhoz)
        self._cursor_pos = (wx, wy)
        if self._poly_mode:
            self.update()

        # ── Hover kurzor visszajelzés ─────────────────────────────────────────
        if not self._rpan_active and not (event.buttons() & Qt.MouseButton.LeftButton):
            hit  = self._hit_test(wx, wy)
            rect = self._display_rect()
            if self._poly_mode:
                self.setCursor(Qt.CursorShape.CrossCursor)
            elif hit >= 0:
                self.setCursor(Qt.CursorShape.PointingHandCursor)
            elif rect is not None and rect.contains(wx, wy):
                self.setCursor(Qt.CursorShape.CrossCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)

        # ── Jobb gombos pásztázás ─────────────────────────────────────────────
        if self._rpan_active and (event.buttons() & Qt.MouseButton.RightButton):
            dx = wx - self._rpan_lx
            dy = wy - self._rpan_ly
            if abs(dx) > 1 or abs(dy) > 1:
                self._rpan_moved = True
                self._pan_by_pixels(dx, dy)
                self._rpan_lx = wx
                self._rpan_ly = wy
            return

        # ── Gumiszalag frissítése ─────────────────────────────────────────────
        if self._rband_p0 is not None and (event.buttons() & Qt.MouseButton.LeftButton):
            dist = math.hypot(wx - self._rband_p0[0], wy - self._rband_p0[1])
            if dist >= self._RBAND_MIN:
                self._rband_active = True
            if self._rband_active:
                self._rband_p1 = (wx, wy)
                self.update()
                return   # pont-húzás NEM fut gumiszalag közben

        # ── Bal gombos pont-húzás ─────────────────────────────────────────────
        if self._drag_idx >= 0 and (event.buttons() & Qt.MouseButton.LeftButton):
            coords = self._w2i(wx, wy)
            if coords is not None:
                ix, iy = coords
                self._dragging = True
                if 0 <= self._drag_idx < len(self._points):
                    self._points[self._drag_idx] = (ix, iy)
                    self.update()
                self.point_moved.emit(self._drag_idx, ix, iy)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if self._rband_active:
                # Gumiszalag lezárása (húzás volt)
                self._finalize_rband()
            elif self._rband_p0 is not None and self._rband_ctrl:
                # Ctrl+klikk húzás nélkül → sokszög csúcs hozzáadása
                wx, wy = self._rband_p0
                if self._poly_mode and self._poly_pts:
                    # Dupla-klikk miatt ne adjunk hozzá azonos pozíciójú pontot
                    lx, ly = self._poly_pts[-1]
                    if math.hypot(wx - lx, wy - ly) > 5:
                        self._poly_pts.append((wx, wy))
                else:
                    # Első csúcs → sokszög mód indítása
                    self._poly_mode = True
                    self._poly_pts  = [(wx, wy)]
            self._rband_active = False
            self._rband_p0     = None
            self._rband_p1     = None
            self._drag_idx     = -1
            self._dragging     = False
            self.update()

        if event.button() == Qt.MouseButton.RightButton:
            self._rpan_active = False
            self._rpan_moved  = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

        super().mouseReleaseEvent(event)

    def _finalize_rband(self) -> None:
        """
        Gumiszalag elengedésekor:
        - normál keret → multi-kijelölés (selection_changed jel)
        - Ctrl+keret   → ROI-keresés (roi_search_requested jel, képkoordinátában)
        """
        if self._rband_p0 is None or self._rband_p1 is None:
            return
        x0, y0 = self._rband_p0
        x1, y1 = self._rband_p1
        rx0, rx1 = min(x0, x1), max(x0, x1)
        ry0, ry1 = min(y0, y1), max(y0, y1)

        # Pontok a keretben
        inside = []
        for i, (ix, iy) in enumerate(self._points):
            wpt = self._i2w(ix, iy)
            if wpt is None:
                continue
            px, py = wpt
            if rx0 <= px <= rx1 and ry0 <= py <= ry1:
                inside.append(i)

        if self._rband_ctrl:
            # Ctrl-keret: 4 sarkos sokszöget küldünk (widget-koordinátában)
            poly = [
                (rx0, ry0), (rx1, ry0),
                (rx1, ry1), (rx0, ry1),
            ]
            self._display_poly = poly
            self._poly_mode    = False
            self._poly_pts     = []
            self.update()
            self.roi_ready.emit(poly)
        else:
            # Normál keret: multi-kijelölés
            self._selected_set = set(inside)
            if inside:
                self.point_selected.emit(inside[0])
            self.selection_changed.emit(inside)

    def _pan_by_pixels(self, dx: float, dy: float) -> None:
        rect = self._display_rect()
        if rect is None or self._img_size is None:
            return
        iw, ih = self._img_size
        eff_scale = rect.width() / iw
        self._pan_cx -= dx / eff_scale
        self._pan_cy -= dy / eff_scale
        self.update()

    def wheelEvent(self, event) -> None:
        """
        Görgetés (Ctrl nélkül) → zoom az aktív pont (vagy egér) körül.
        Ctrl+görgetés           → pontindex léptetése (fel = kisebb, le = nagyobb).
        """
        delta = event.angleDelta().y()

        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # ── Ctrl+görgetés: pontindex léptetés ────────────────────────────
            n = len(self._points)
            if n == 0:
                event.accept()
                return
            cur = self._selected if 0 <= self._selected < n else 0
            new_idx = (cur - 1) % n if delta > 0 else (cur + 1) % n
            self.point_selected.emit(new_idx)
            event.accept()

        else:
            # ── Görgetés: zoom ────────────────────────────────────────────────
            if self._img_size is None:
                event.accept()
                return
            factor   = self._ZOOM_STEP if delta > 0 else 1.0 / self._ZOOM_STEP
            new_zoom = max(self._ZOOM_MIN, min(self._ZOOM_MAX, self._zoom * factor))
            if new_zoom == self._zoom:
                event.accept()
                return
            # Zoom középpontja: aktív pont > egér > pan-centrum
            if 0 <= self._selected < len(self._points):
                fx, fy = self._points[self._selected]
            else:
                mx  = float(event.position().x())
                my  = float(event.position().y())
                hit = self._w2i(mx, my)
                fx, fy = hit if hit else (self._pan_cx, self._pan_cy)
            ratio        = self._zoom / new_zoom
            self._pan_cx = fx + (self._pan_cx - fx) * ratio
            self._pan_cy = fy + (self._pan_cy - fy) * ratio
            self._zoom   = new_zoom
            self.update()
            event.accept()

    def keyPressEvent(self, event) -> None:
        """
        Space         → zoom szinkron a másik canvasra
        Ctrl+0        → zoom visszaállítás
        Delete        → kijelölt pont(ok) törlése (egyszeres vagy multi)
        Ctrl+Z        → visszavonás kérése
        Nyílbillentyű → aktív pont mozgatása 1 px (Shift = 10 px)
        """
        key = event.key()

        if key == Qt.Key.Key_Space and self._img_size is not None:
            iw, ih = self._img_size
            self.zoom_sync_requested.emit(
                self._zoom, self._pan_cx / iw, self._pan_cy / ih)
            event.accept()

        elif key == Qt.Key.Key_0 and (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            self._reset_zoom()
            event.accept()

        elif key == Qt.Key.Key_Delete:
            if self._selected_set:
                self.points_delete_multi_requested.emit()
            elif self._selected >= 0:
                self.point_deleted.emit(self._selected)
            event.accept()

        elif key == Qt.Key.Key_Z and (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            self.undo_requested.emit()
            event.accept()

        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            # Enter → sokszög lezárása (ha aktív és van elég csúcs)
            if self._poly_mode and len(self._poly_pts) >= 3:
                self._close_polygon()
            event.accept()

        elif key == Qt.Key.Key_Escape:
            # Escape → sokszög megszakítása
            if self._poly_mode:
                self._poly_mode    = False
                self._poly_pts     = []
                self._display_poly = []
                self.update()
            event.accept()

        elif key in (Qt.Key.Key_Left, Qt.Key.Key_Right,
                     Qt.Key.Key_Up,   Qt.Key.Key_Down):
            if 0 <= self._selected < len(self._points) and self._img_size is not None:
                step = 10.0 if (event.modifiers() & Qt.KeyboardModifier.ShiftModifier) else 1.0
                ix, iy = self._points[self._selected]
                if   key == Qt.Key.Key_Left:  ix -= step
                elif key == Qt.Key.Key_Right: ix += step
                elif key == Qt.Key.Key_Up:    iy -= step
                elif key == Qt.Key.Key_Down:  iy += step
                iw, ih = self._img_size
                ix = max(0.0, min(float(iw - 1), ix))
                iy = max(0.0, min(float(ih - 1), iy))
                self._points[self._selected] = (ix, iy)
                self.update()
                self.point_moved.emit(self._selected, ix, iy)
            event.accept()

        else:
            super().keyPressEvent(event)

    def _reset_zoom(self) -> None:
        if self._img_size is None:
            return
        iw, ih = self._img_size
        self._zoom   = 1.0
        self._pan_cx = iw / 2.0
        self._pan_cy = ih / 2.0
        self.update()

    def set_zoom_view(self, zoom: float, rel_x: float, rel_y: float) -> None:
        if self._img_size is None:
            return
        iw, ih       = self._img_size
        self._zoom   = max(self._ZOOM_MIN, min(self._ZOOM_MAX, zoom))
        self._pan_cx = rel_x * iw
        self._pan_cy = rel_y * ih
        self.update()

    def _show_context_menu(self, index: int, global_pos: QPoint) -> None:
        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu{background:#252a33;color:#eee;border:1px solid #444;}"
            "QMenu::item:selected{background:#3a3f4a;}"
            "QMenu::separator{background:#444;height:1px;margin:3px 6px;}"
        )
        header = QAction(f"  Pont  #{index + 1}", self)
        header.setEnabled(False)
        menu.addAction(header)
        menu.addSeparator()

        act_sel = QAction("Kiválasztás", self)
        act_sel.triggered.connect(lambda: self.point_selected.emit(index))
        menu.addAction(act_sel)

        act_del = QAction("Törlés  (párjával együtt)", self)
        act_del.triggered.connect(lambda: self.point_deleted.emit(index))
        menu.addAction(act_del)

        menu.exec(global_pos)

    # ── Rajzolás – paintEvent ─────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Háttér + keret
        painter.fillRect(self.rect(), _C_BG)
        painter.setPen(QPen(_C_BORDER, 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))

        if self._pixmap_orig is None:
            painter.setPen(QPen(_C_PLACEHOLDER, 1))
            font = QFont()
            font.setPointSize(11)
            painter.setFont(font)
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter,
                f"{self.title}\n\nNincs kép betöltve\n\nDupla klikk: pont hozzáadása"
            )
            return

        # Kép kirajzolása
        rect = self._display_rect()
        if rect:
            scaled = self._pixmap_orig.scaled(
                int(rect.width()), int(rect.height()),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            painter.drawPixmap(int(rect.left()), int(rect.top()), scaled)

        # Pontok
        font = QFont()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)

        for i, (ix, iy) in enumerate(self._points):
            wpt = self._i2w(ix, iy)
            if wpt is None:
                continue
            px, py = wpt
            active = (i == self._selected)
            multi  = (i in self._selected_set)

            if active:
                color, fill, r = _C_ACTIVE, _C_FILL_A, _R_ACTIVE
            elif multi:
                color, fill, r = _C_MULTI, _C_FILL_M, _R_MULTI
            else:
                color, fill, r = _C_NORMAL, _C_FILL_N, _R_NORMAL

            painter.setPen(QPen(color, 2))
            painter.setBrush(QBrush(fill))
            painter.drawEllipse(int(px) - r, int(py) - r, r * 2, r * 2)

            if active:
                painter.setPen(QPen(color, 1))
                ext = r + 5
                painter.drawLine(int(px) - ext, int(py), int(px) + ext, int(py))
                painter.drawLine(int(px), int(py) - ext, int(px), int(py) + ext)

            txt_color = _C_TEXT_A if active else (_C_MULTI if multi else _C_TEXT)
            painter.setPen(QPen(txt_color, 1))
            painter.drawText(int(px) + r + 3, int(py) + 4, str(i + 1))

        # Gumiszalag-keret
        if self._rband_active and self._rband_p0 and self._rband_p1:
            x0, y0 = self._rband_p0
            x1, y1 = self._rband_p1
            rx0, rx1 = min(x0, x1), max(x0, x1)
            ry0, ry1 = min(y0, y1), max(y0, y1)
            c_line = _C_RBAND_ROI if self._rband_ctrl else _C_RBAND_SEL
            c_fill = QColor(c_line.red(), c_line.green(), c_line.blue(), 28)
            painter.setPen(QPen(c_line, 1, Qt.PenStyle.DashLine))
            painter.setBrush(QBrush(c_fill))
            painter.drawRect(int(rx0), int(ry0), int(rx1 - rx0), int(ry1 - ry0))

        # ── Sokszög-ROI rajzolás ─────────────────────────────────────────────

        def _draw_poly(pts, color, closed=False):
            if len(pts) < 1:
                return
            pen = QPen(color, 2, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.setBrush(QBrush(_C_POLY_FILL))
            # Élek
            for k in range(len(pts) - 1):
                ax, ay = pts[k]
                bx, by = pts[k + 1]
                painter.drawLine(int(ax), int(ay), int(bx), int(by))
            if closed and len(pts) >= 3:
                ax, ay = pts[-1]
                bx, by = pts[0]
                painter.drawLine(int(ax), int(ay), int(bx), int(by))
            # Csúcsok
            for k, (px, py) in enumerate(pts):
                if k == 0:
                    r = 6
                    painter.setBrush(QBrush(color))
                else:
                    r = 4
                    painter.setBrush(QBrush(_C_POLY_FILL))
                painter.drawEllipse(int(px) - r, int(py) - r, r * 2, r * 2)

        if self._poly_mode and self._poly_pts:
            # Rajzolás alatt: folyamatos élek + preview vonal az egér felé
            _draw_poly(self._poly_pts, _C_POLY_DRAW, closed=False)
            if len(self._poly_pts) >= 1:
                lx, ly = self._poly_pts[-1]
                cx, cy = self._cursor_pos
                painter.setPen(QPen(_C_POLY_DRAW, 1, Qt.PenStyle.DashLine))
                painter.drawLine(int(lx), int(ly), int(cx), int(cy))
            # Segítség szöveg
            painter.setPen(QPen(QColor("#FFB300"), 1))
            font2 = QFont()
            font2.setPointSize(9)
            painter.setFont(font2)
            n = len(self._poly_pts)
            hint = (f"Ctrl+klikk: csúcs ({n} db)  |  Enter / Ctrl+2×klikk: lezár"
                    "  |  Esc: mégse")
            painter.drawText(6, self.height() - 8, hint)

        elif self._display_poly:
            # Lezárt sokszög: menü megjelenéséig látható
            _draw_poly(self._display_poly, _C_POLY_DONE, closed=True)


# ────────────────────────────────────────────────────────────────────────────
#  PointEditorWidget
# ────────────────────────────────────────────────────────────────────────────
class PointEditorWidget(QWidget):
    """
    Kétpaneles pontszerkesztő (Kép A + Kép B).

    Párosítási logika: anchor_points_a[i] és anchor_points_b[i] egy pár.
    Dupla kattintáskor a párpont ugyanarra az arányos képpozícióra kerül
    a másik képen (pl. kép A 30%-os x-pozíciója → kép B 30%-os x-pozíciója).

    Billentyűk (canvas fókuszban)
    -----------------------------
    Dupla bal klikk       → pont hozzáadása
    Bal húzás (képen)     → gumiszalag-kijelölés (kék keret)
    Ctrl+bal húzás        → ROI-keresés indítása (sárga keret)
    Delete                → kijelölt pont(ok) törlése
    Space                 → zoom szinkron a másik canvasra
    Ctrl+0                → zoom visszaállítás
    Ctrl+görgetés         → pontindex léptetés
    Görgetés              → zoom
    Ctrl+Z                → visszavonás
    Nyílbillentyűk        → pont mozgatása 1 px (Shift = 10 px)

    Publikus API
    ------------
    load_images()                 – képek és pontok betöltése
    refresh_from_project()        – csak a pontlista frissítése
    undo()                        – visszavonás
    points_changed (signal)       – bármilyen változás után
    roi_search_requested (signal) – (x1,y1,x2,y2,side) ROI keresés kérve
    """

    points_changed       = pyqtSignal()
    # (img_polygon, side, delete_in_roi, backend)
    roi_search_requested = pyqtSignal(list, str, bool, str)

    _UNDO_LIMIT = 50

    _MENU_STYLE = (
        "QMenu{background:#252a33;color:#eee;border:1px solid #444;}"
        "QMenu::item{padding:4px 20px 4px 24px;}"
        "QMenu::item:selected{background:#3a3f4a;}"
        "QMenu::item:disabled{color:#555;}"
        "QMenu::separator{background:#444;height:1px;margin:3px 6px;}"
        "QMenu::indicator{width:14px;height:14px;}"
    )
    _BACKENDS = [
        "SuperPoint + LightGlue",
        "DISK + LightGlue",
        "LoFTR (kornia)",
        "SIFT (OpenCV)",
    ]

    def __init__(self, project) -> None:
        super().__init__()
        self.project            = project
        self._selected:          int      = -1
        self._selected_set:      Set[int] = set()
        self._undo_stack:        List[Tuple[List, List]] = []
        self._roi_last_backend:  str  = "SuperPoint + LightGlue"
        self._roi_delete_in_roi: bool = False
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 4, 0, 0)
        root.setSpacing(4)

        # ── Eszköztár ────────────────────────────────────────────────────────
        bar = QHBoxLayout()
        bar.setSpacing(6)

        self.btn_del = QPushButton("✕  Kiválasztott pár(ok) törlése")
        self.btn_del.setEnabled(False)
        self.btn_del.setFixedHeight(28)
        self.btn_del.setStyleSheet(
            "QPushButton{background:#6b1e1e;color:#eee;padding:0 12px;"
            "border-radius:4px;font-size:12px;}"
            "QPushButton:hover{background:#922828;}"
            "QPushButton:disabled{background:#2e2e2e;color:#555;}"
        )
        self.btn_del.clicked.connect(self._delete_selection)

        self.lbl_hint = QLabel(
            "2× klikk: pont  |  Húzás: kijelölés  |  "
            "Ctrl+klikk: sokszög-ROI  |  Ctrl+húzás: téglalap-ROI  |  "
            "Enter: ROI lezár  |  Esc: mégse  |  "
            "Delete: törlés  |  Ctrl+Görgetés: pontváltás  |  Görgetés: zoom"
        )
        self.lbl_hint.setStyleSheet("color:#666;font-size:11px;")

        self.lbl_count = QLabel("Pontpárok: 0")
        self.lbl_count.setStyleSheet("color:#aaa;font-size:12px;")

        bar.addWidget(self.btn_del)
        bar.addSpacing(8)
        bar.addWidget(self.lbl_hint)
        bar.addStretch()
        bar.addWidget(self.lbl_count)

        # ── Két canvas ───────────────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.canvas_a = PointEditorCanvas("Kép  A")
        self.canvas_b = PointEditorCanvas("Kép  B")
        splitter.addWidget(self.canvas_a)
        splitter.addWidget(self.canvas_b)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        root.addLayout(bar)
        root.addWidget(splitter, stretch=1)

        # ── Szignálok ────────────────────────────────────────────────────────
        self.canvas_a.point_added.connect(self._added_a)
        self.canvas_b.point_added.connect(self._added_b)
        self.canvas_a.point_moved.connect(self._moved_a)
        self.canvas_b.point_moved.connect(self._moved_b)
        self.canvas_a.point_deleted.connect(self._delete_pair)
        self.canvas_b.point_deleted.connect(self._delete_pair)
        self.canvas_a.point_selected.connect(self._select)
        self.canvas_b.point_selected.connect(self._select)
        # Multi-kijelölés szinkron
        self.canvas_a.selection_changed.connect(self._on_selection_a)
        self.canvas_b.selection_changed.connect(self._on_selection_b)
        # Multi-törlés
        self.canvas_a.points_delete_multi_requested.connect(self._delete_selection)
        self.canvas_b.points_delete_multi_requested.connect(self._delete_selection)
        # ROI keresés (roi_ready → helyi menü → roi_search_requested)
        self.canvas_a.roi_ready.connect(
            lambda poly: self._show_roi_menu(poly, "A"))
        self.canvas_b.roi_ready.connect(
            lambda poly: self._show_roi_menu(poly, "B"))
        # Zoom szinkron
        self.canvas_a.zoom_sync_requested.connect(self.canvas_b.set_zoom_view)
        self.canvas_b.zoom_sync_requested.connect(self.canvas_a.set_zoom_view)
        # Középső gomb + Ctrl+Z
        self.canvas_a.middle_clicked.connect(self._on_middle_click)
        self.canvas_b.middle_clicked.connect(self._on_middle_click)
        self.canvas_a.undo_requested.connect(self._pop_undo)
        self.canvas_b.undo_requested.connect(self._pop_undo)

    # ── Publikus API ─────────────────────────────────────────────────────────

    def load_images(self) -> None:
        if self.project.image_a is not None:
            self.canvas_a.set_image(self.project.image_a)
        if self.project.image_b is not None:
            self.canvas_b.set_image(self.project.image_b)
        self._sync()

    def refresh_from_project(self) -> None:
        self._sync()

    def undo(self) -> bool:
        return self._pop_undo()

    # ── Szinkronizáció ───────────────────────────────────────────────────────

    def _sync(self) -> None:
        pts_a = [tuple(p) for p in self.project.anchor_points_a]
        pts_b = [tuple(p) for p in self.project.anchor_points_b]
        self.canvas_a.set_points(pts_a)
        self.canvas_b.set_points(pts_b)
        self.canvas_a.set_selected(self._selected)
        self.canvas_b.set_selected(self._selected)
        # Érvénytelen multi-indexek eltávolítása
        n = min(len(pts_a), len(pts_b))
        self._selected_set = {i for i in self._selected_set if i < n}
        self.canvas_a.set_selected_set(list(self._selected_set))
        self.canvas_b.set_selected_set(list(self._selected_set))
        self.lbl_count.setText(f"Pontpárok: {n}")

    def _select(self, index: int) -> None:
        self._selected = index
        self.canvas_a.set_selected(index)
        self.canvas_b.set_selected(index)
        has_sel = index >= 0 or bool(self._selected_set)
        self.btn_del.setEnabled(has_sel)

    def _on_selection_a(self, indices: List[int]) -> None:
        """Canvas A gumiszalag → szinkronizálás canvas B-re."""
        self._selected_set = set(indices)
        self.canvas_b.set_selected_set(indices)
        self.btn_del.setEnabled(bool(indices) or self._selected >= 0)

    def _on_selection_b(self, indices: List[int]) -> None:
        """Canvas B gumiszalag → szinkronizálás canvas A-ra."""
        self._selected_set = set(indices)
        self.canvas_a.set_selected_set(indices)
        self.btn_del.setEnabled(bool(indices) or self._selected >= 0)

    # ── Undo stack ───────────────────────────────────────────────────────────

    def _push_undo(self) -> None:
        import copy
        state = (
            copy.deepcopy(self.project.anchor_points_a),
            copy.deepcopy(self.project.anchor_points_b),
        )
        self._undo_stack.append(state)
        if len(self._undo_stack) > self._UNDO_LIMIT:
            self._undo_stack.pop(0)

    def _pop_undo(self) -> bool:
        if not self._undo_stack:
            return False
        pts_a, pts_b = self._undo_stack.pop()
        self.project.anchor_points_a[:] = pts_a
        self.project.anchor_points_b[:] = pts_b
        new_sel = min(self._selected, len(pts_a) - 1)
        self._selected_set.clear()
        self._select(new_sel)
        self._sync()
        self.points_changed.emit()
        return True

    # ── ROI helyi menü ───────────────────────────────────────────────────────

    def _show_roi_menu(self, widget_poly: list, side: str) -> None:
        """
        Megjelenik a ROI-keresés helyi menüje.

        widget_poly : a sokszög csúcsai widget-koordinátában
        side        : "A" vagy "B" – melyik canvason rajzolták
        """
        canvas = self.canvas_a if side == "A" else self.canvas_b

        # Widget → kép koordináta konverzió
        img_poly = []
        for wx, wy in widget_poly:
            ic = canvas._w2i(wx, wy)
            if ic is not None:
                img_poly.append([float(ic[0]), float(ic[1])])
            else:
                # Képen kívüli csúcs: a kép szélére klampoljuk
                rect = canvas._display_rect()
                iw, ih = (canvas._img_size or (1, 1))
                if rect:
                    cx = max(rect.left(), min(rect.right(),  wx))
                    cy = max(rect.top(),  min(rect.bottom(), wy))
                    rx = (cx - rect.left()) / rect.width()
                    ry = (cy - rect.top())  / rect.height()
                    img_poly.append([rx * iw, ry * ih])

        if len(img_poly) < 3:
            canvas._display_poly = []
            canvas.update()
            return

        # Sokszög megjelenítése amíg a menü nyitva van
        canvas._display_poly = list(widget_poly)
        canvas.update()

        # ── Menü felépítése ───────────────────────────────────────────────────
        menu = QMenu(canvas)
        menu.setStyleSheet(self._MENU_STYLE)

        # Cím (nem kattintható)
        n_verts    = len(img_poly)
        shape_name = "téglalap-ROI" if n_verts == 4 else f"{n_verts}-szög ROI"
        title_act  = QAction(f"  {shape_name}  –  keresési beállítások", menu)
        title_act.setEnabled(False)
        menu.addAction(title_act)
        menu.addSeparator()

        # Backend almenü
        bk_menu  = menu.addMenu("▸  Backend: " + self._roi_last_backend)
        bk_menu.setStyleSheet(self._MENU_STYLE)
        bk_group = QActionGroup(bk_menu)
        bk_group.setExclusive(True)
        bk_acts  = {}
        for b in self._BACKENDS:
            act = QAction(b, bk_menu)
            act.setCheckable(True)
            act.setChecked(b == self._roi_last_backend)
            bk_group.addAction(act)
            bk_menu.addAction(act)
            bk_acts[act] = b

        menu.addSeparator()

        # Meglévő pontok törlése ROI-ban (be/ki kapcsolható)
        del_act = QAction("Meglévő pontok törlése a területen belül", menu)
        del_act.setCheckable(True)
        del_act.setChecked(self._roi_delete_in_roi)
        del_act.setToolTip(
            "Ha be van jelölve, keresés előtt törli az összes meglévő\n"
            "pontpárt amelyek bármelyik tagja a berajzolt területen belül esik.")
        menu.addAction(del_act)
        menu.addSeparator()

        # Keresés gomb
        search_act = QAction(f"▶  ROI keresés indítása", menu)
        menu.addAction(search_act)

        cancel_act = QAction("✕  Mégse", menu)
        menu.addAction(cancel_act)

        # ── Menü megjelenítése ────────────────────────────────────────────────
        last_wx, last_wy = widget_poly[-1]
        global_pos = canvas.mapToGlobal(QPoint(int(last_wx), int(last_wy)))
        chosen     = menu.exec(global_pos)

        # Backend-csere: frissítjük az állapotot, majd újra megjelenítjük a menüt
        if chosen in bk_acts:
            self._roi_last_backend = bk_acts[chosen]
            self._show_roi_menu(widget_poly, side)  # rekurzív újranyitás
            return

        # Delete toggle: frissítjük az állapotot, majd újra megjelenítjük
        if chosen == del_act:
            self._roi_delete_in_roi = del_act.isChecked()
            self._show_roi_menu(widget_poly, side)
            return

        # Menü bezárása (bármelyik ágon): display_poly törlése
        canvas._display_poly = []
        canvas.update()

        if chosen == search_act:
            self._roi_delete_in_roi = del_act.isChecked()
            self.roi_search_requested.emit(
                img_poly, side, self._roi_delete_in_roi, self._roi_last_backend)

    def _on_middle_click(self) -> None:
        self.canvas_a._reset_zoom()
        self.canvas_b._reset_zoom()
        self._pop_undo()

    # ── Pont hozzáadása ──────────────────────────────────────────────────────

    def _added_a(self, x: float, y: float) -> None:
        self._push_undo()
        self.project.anchor_points_a.append([x, y])
        bx, by = self._mirror_a_to_b(x, y)
        self.project.anchor_points_b.append([bx, by])
        self._selected_set.clear()
        self._select(len(self.project.anchor_points_a) - 1)
        self._sync()
        self.points_changed.emit()

    def _added_b(self, x: float, y: float) -> None:
        self._push_undo()
        self.project.anchor_points_b.append([x, y])
        ax, ay = self._mirror_b_to_a(x, y)
        self.project.anchor_points_a.append([ax, ay])
        self._selected_set.clear()
        self._select(len(self.project.anchor_points_b) - 1)
        self._sync()
        self.points_changed.emit()

    def _mirror_a_to_b(self, x: float, y: float) -> Tuple[float, float]:
        ia, ib = self.project.image_a, self.project.image_b
        if ia is None or ib is None:
            return x, y
        ha, wa = ia.shape[:2]
        hb, wb = ib.shape[:2]
        return (x / wa) * wb, (y / ha) * hb

    def _mirror_b_to_a(self, x: float, y: float) -> Tuple[float, float]:
        ia, ib = self.project.image_a, self.project.image_b
        if ia is None or ib is None:
            return x, y
        ha, wa = ia.shape[:2]
        hb, wb = ib.shape[:2]
        return (x / wb) * wa, (y / hb) * ha

    # ── Pont mozgatása ───────────────────────────────────────────────────────

    def _moved_a(self, index: int, x: float, y: float) -> None:
        if 0 <= index < len(self.project.anchor_points_a):
            self.project.anchor_points_a[index] = [x, y]
            self.points_changed.emit()

    def _moved_b(self, index: int, x: float, y: float) -> None:
        if 0 <= index < len(self.project.anchor_points_b):
            self.project.anchor_points_b[index] = [x, y]
            self.points_changed.emit()

    # ── Pont törlése ─────────────────────────────────────────────────────────

    def _delete_pair(self, index: int) -> None:
        """Egyszeres törlés (jobb klikk menü vagy Delete egyszeres kijelöléssel)."""
        self._push_undo()
        pa = self.project.anchor_points_a
        pb = self.project.anchor_points_b
        if 0 <= index < len(pa):
            pa.pop(index)
        if 0 <= index < len(pb):
            pb.pop(index)
        self._selected_set.discard(index)
        self._selected_set = {i if i < index else i - 1
                               for i in self._selected_set if i != index}
        new_sel = min(self._selected, len(pa) - 1)
        self._select(new_sel)
        self._sync()
        self.points_changed.emit()

    def _delete_selection(self) -> None:
        """
        Multi-törlés: ha van multi-kijelölés, azokat törli.
        Ha nincs, az egyszeresen kijelöltet törli.
        """
        if self._selected_set:
            self._push_undo()
            pa = self.project.anchor_points_a
            pb = self.project.anchor_points_b
            # Csökkenő indexsorrendben töröl, hogy az indexek ne csússzanak el
            for idx in sorted(self._selected_set, reverse=True):
                if 0 <= idx < len(pa):
                    pa.pop(idx)
                if 0 <= idx < len(pb):
                    pb.pop(idx)
            self._selected_set.clear()
            new_sel = min(self._selected, len(pa) - 1)
            self._select(new_sel)
            self._sync()
            self.points_changed.emit()
        elif self._selected >= 0:
            self._delete_pair(self._selected)
