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

from PyQt6.QtCore import Qt, QPointF, QRectF, QPoint, QSize, pyqtSignal
from PyQt6.QtGui import (
    QAction, QActionGroup, QBrush, QColor, QFont,
    QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF,
)
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QMenu, QPushButton, QSpinBox,
    QSizePolicy, QSplitter, QVBoxLayout, QWidget,
)

try:
    from TRANSLATIONS import tr
except ImportError:
    def tr(text: str) -> str: return text

# ── Konfiguráció betöltése ────────────────────────────────────────────────────
# Az archmorph_config.toml fájlból olvassa az értékeket (ha létezik).
# Ha nem létezik, vagy hibás, az alábbi alapértelmezett értékek maradnak.
try:
    from archmorph_config_loader import cfg, cfg_rgba
except ImportError:
    # Ha a config_loader nincs meg, semleges fallback
    def cfg(key, default):      return default        # type: ignore[misc]
    def cfg_rgba(key, default): return default        # type: ignore[misc]

# ── Vizuális konstansok (archmorph_config.toml → [ui.sizes] és [ui.colors]) ──

_R_NORMAL = cfg("ui.sizes.point_radius_normal", 6)    # Normál pont sugara (px)
_R_ACTIVE = cfg("ui.sizes.point_radius_active", 10)   # Aktív pont sugara (px)
_R_MULTI  = cfg("ui.sizes.point_radius_multi",  8)    # Multi-kijelölt pont sugara (px)
_HIT_R    = cfg("ui.sizes.point_hit_radius",    14)   # Kattintási érzékenység (px)

_C_NORMAL = QColor(cfg("ui.colors.points.normal",      "#ff8a3d"))
_C_ACTIVE = QColor(cfg("ui.colors.points.active",      "#ff3333"))
_C_MULTI  = QColor(cfg("ui.colors.points.multi",       "#44aaff"))
_C_TEXT   = QColor(cfg("ui.colors.points.text",        "#ffffff"))
_C_TEXT_A = QColor(cfg("ui.colors.points.text_active", "#ffcccc"))
_C_BG     = QColor(cfg("ui.colors.canvas.background",  "#1a1f24"))
_C_BORDER = QColor(cfg("ui.colors.canvas.border",      "#343b45"))
_C_PLACEHOLDER = QColor(cfg("ui.colors.canvas.placeholder", "#4a5260"))

# Pontok belső kitöltése (RGBA)
_C_FILL_N = QColor(*cfg_rgba("ui.colors.points.fill.normal_rgba", (255, 138,  61,  70)))
_C_FILL_A = QColor(*cfg_rgba("ui.colors.points.fill.active_rgba", (255,  51,  51, 150)))
_C_FILL_M = QColor(*cfg_rgba("ui.colors.points.fill.multi_rgba",  ( 68, 170, 255,  90)))

# Gumiszalag stílus
_C_RBAND_SEL = QColor(cfg("ui.colors.rband.selection", "#88bbff"))
_C_RBAND_ROI = QColor(cfg("ui.colors.rband.roi",       "#ffcc44"))

# Sokszög-ROI stílus
_C_POLY_DRAW = QColor(cfg("ui.colors.polygon.draw", "#FFB300"))
_C_POLY_DONE = QColor(cfg("ui.colors.polygon.done", "#FF6F00"))
_C_POLY_FILL = QColor(*cfg_rgba("ui.colors.polygon.fill_rgba", (255, 179, 0, 35)))

# Tartós téglalap-ROI stílus
_C_ROI_BORDER  = QColor(cfg("ui.colors.roi.border", "#ff9900"))
_C_ROI_HANDLE  = QColor(cfg("ui.colors.roi.handle", "#ffffff"))
_C_ROI_FILL    = QColor(*cfg_rgba("ui.colors.roi.fill_rgba",    (255, 153,   0,  22)))
_C_ROI_OUTSIDE = QColor(*cfg_rgba("ui.colors.roi.outside_rgba", (  0,   0,   0,  90)))
_ROI_HANDLE_R  = cfg("ui.sizes.roi_handle_radius", 6)    # fogópontok sugara (px)
_ROI_MIN_SIZE  = cfg("ui.sizes.roi_min_size",       10)   # minimum ROI méret (px)


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
    curve_done                  = pyqtSignal(list)   # [(x,y)…] képkoordban
    point_deleted               = pyqtSignal(int)
    points_delete_multi_requested = pyqtSignal()
    point_selected              = pyqtSignal(int)
    selection_changed           = pyqtSignal(list)
    roi_ready                   = pyqtSignal(list)   # sokszög widget-koordinátákban
    roi_rect_changed            = pyqtSignal(object) # QRectF képkoordban vagy None
    zoom_sync_requested         = pyqtSignal(float, float, float)
    middle_clicked              = pyqtSignal()
    undo_requested              = pyqtSignal()
    image_drop_requested        = pyqtSignal(str)    # kép drag & drop

    _IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

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

        # Tartós téglalap-ROI (képkoordban tárolva)
        self._roi_img:       Optional[QRectF]  = None
        self._roi_mode:      str               = ""    # "" | "move" | handle-neve
        self._roi_drag_wx:   float             = 0.0
        self._roi_drag_wy:   float             = 0.0
        self._roi_img_start: Optional[QRectF]  = None

        # Drag & drop
        self._drag_hover: bool = False
        self.setAcceptDrops(True)

        # Vonallánc / ív rajzolás
        self._draw_mode:  str                      = "point"  # "point"|"polyline"|"arc"
        self._curve_pts:  List[Tuple[float,float]] = []       # csúcsok képkoordban

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

    # ── Tartós téglalap-ROI – publikus API ──────────────────────────────────

    def set_roi_from_image(self, rect: Optional[QRectF]) -> None:
        """ROI beállítása képkoordináták alapján (a másik canvasról szinkronizálva)."""
        self._roi_img = QRectF(rect) if rect is not None else None
        self.update()

    def get_roi_image(self) -> Optional[QRectF]:
        """Visszaadja a ROI-t képkoordinátában (None ha nincs)."""
        return QRectF(self._roi_img) if self._roi_img is not None else None

    def clear_roi(self) -> None:
        self._roi_img   = None
        self._roi_mode  = ""
        self.update()
        self.roi_rect_changed.emit(None)

    # ── Vonallánc / ív rajzolás – publikus API ───────────────────────────────

    def set_draw_mode(self, mode: str) -> None:
        """'point' | 'polyline' | 'arc' – módváltáskor töröl minden félkész görbét."""
        if mode != self._draw_mode:
            self._draw_mode = mode
            self._curve_pts = []
            self.update()

    def cancel_curve(self) -> None:
        """Félkész görbét töröl (pl. ha a másik canvas is törli)."""
        self._curve_pts = []
        self.update()

    def _finish_curve(self) -> None:
        """Görbe lezárása: curve_done jel küldése, ha van elég csúcs."""
        pts = list(self._curve_pts)
        self._curve_pts = []
        self.update()
        if len(pts) >= 2:
            self.curve_done.emit(pts)

    # ── Tartós téglalap-ROI – belső segédek ─────────────────────────────────

    def _roi_widget_rect(self) -> Optional[QRectF]:
        """ROI képkoordból widget-koordinátára."""
        if self._roi_img is None:
            return None
        tl = self._i2w(self._roi_img.left(),  self._roi_img.top())
        br = self._i2w(self._roi_img.right(), self._roi_img.bottom())
        if tl is None or br is None:
            return None
        return QRectF(QPointF(*tl), QPointF(*br)).normalized()

    _HANDLE_NAMES = ("tl", "tc", "tr", "ml", "mr", "bl", "bc", "br")

    def _roi_handles_w(self) -> dict:
        """8 fogópont pozíciója widget-koordinátában → {name: (x, y)}."""
        r = self._roi_widget_rect()
        if r is None:
            return {}
        cx, cy = r.center().x(), r.center().y()
        return {
            "tl": (r.left(),   r.top()),
            "tc": (cx,          r.top()),
            "tr": (r.right(),  r.top()),
            "ml": (r.left(),   cy),
            "mr": (r.right(),  cy),
            "bl": (r.left(),   r.bottom()),
            "bc": (cx,          r.bottom()),
            "br": (r.right(),  r.bottom()),
        }

    def _hit_roi_handle(self, wx: float, wy: float) -> str:
        """Melyik fogópontra kattintott? Üres string ha egyik sem."""
        for name, (hx, hy) in self._roi_handles_w().items():
            if math.hypot(wx - hx, wy - hy) <= _ROI_HANDLE_R + 4:
                return name
        return ""

    def _hit_roi_body(self, wx: float, wy: float) -> bool:
        """A ROI belső területén van az egér (fogóponton kívül)?"""
        r = self._roi_widget_rect()
        if r is None:
            return False
        return r.contains(wx, wy) and self._hit_roi_handle(wx, wy) == ""

    _HANDLE_CURSORS = {
        "tl": Qt.CursorShape.SizeFDiagCursor,
        "br": Qt.CursorShape.SizeFDiagCursor,
        "tr": Qt.CursorShape.SizeBDiagCursor,
        "bl": Qt.CursorShape.SizeBDiagCursor,
        "tc": Qt.CursorShape.SizeVerCursor,
        "bc": Qt.CursorShape.SizeVerCursor,
        "ml": Qt.CursorShape.SizeHorCursor,
        "mr": Qt.CursorShape.SizeHorCursor,
    }

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
            # Görbe módban: jobb klikk = utolsó csúcs törlése (vagy mégse)
            if self._draw_mode in ("polyline", "arc"):
                if self._curve_pts:
                    self._curve_pts.pop()
                    self.update()
                return
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
            ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)

            # ── Görbe rajzolás (vonallánc / ív) ─────────────────────────────
            if self._draw_mode in ("polyline", "arc") and not ctrl:
                coords = self._w2i(wx, wy)
                if coords is not None:
                    self._curve_pts.append(coords)
                    # Ívnél 3. csúcs után automatikusan kész
                    if self._draw_mode == "arc" and len(self._curve_pts) == 3:
                        self._finish_curve()
                    else:
                        self.update()
                return

            # ── ROI fogópont / mozgatás ── (Ctrl nélkül, ROI létezik) ────────
            if not ctrl and self._roi_img is not None:
                handle = self._hit_roi_handle(wx, wy)
                if handle:
                    self._roi_mode      = handle
                    self._roi_drag_wx   = wx
                    self._roi_drag_wy   = wy
                    self._roi_img_start = QRectF(self._roi_img)
                    return
                if self._hit_roi_body(wx, wy):
                    self._roi_mode      = "move"
                    self._roi_drag_wx   = wx
                    self._roi_drag_wy   = wy
                    self._roi_img_start = QRectF(self._roi_img)
                    return
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
        Dupla kattintás vonallánc módban → görbe lezárása.
        """
        if event.button() == Qt.MouseButton.LeftButton:
            wx = float(event.position().x())
            wy = float(event.position().y())
            ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)

            # ── Vonallánc mód: dupla klikk = lezárás ────────────────────────
            if self._draw_mode == "polyline" and not ctrl:
                # A Press esemény már hozzáadta az utolsó csúcsot — eltávolítjuk
                if self._curve_pts:
                    self._curve_pts.pop()
                if len(self._curve_pts) >= 2:
                    self._finish_curve()
                else:
                    self._curve_pts = []
                    self.update()
                super().mouseDoubleClickEvent(event)
                return

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

        # Kurzorpozíció frissítése (sokszög + görbe preview vonalhoz)
        self._cursor_pos = (wx, wy)
        if self._poly_mode or (self._draw_mode in ("polyline", "arc") and self._curve_pts):
            self.update()

        # ── ROI drag: mozgatás vagy átméretezés ──────────────────────────────
        if self._roi_mode and (event.buttons() & Qt.MouseButton.LeftButton):
            start_ic = self._w2i(self._roi_drag_wx, self._roi_drag_wy)
            curr_ic  = self._w2i(wx, wy)
            # Ha az egér a kép területén kívülre ment, klampoljuk a képre
            rect = self._display_rect()
            if rect is None or self._roi_img_start is None:
                return
            if curr_ic is None:
                cwx = max(rect.left(), min(wx, rect.right()))
                cwy = max(rect.top(),  min(wy, rect.bottom()))
                curr_ic = self._w2i(cwx, cwy) or (0.0, 0.0)
            if start_ic is None:
                return
            dx = curr_ic[0] - start_ic[0]
            dy = curr_ic[1] - start_ic[1]
            r  = QRectF(self._roi_img_start)
            m  = self._roi_mode
            if m == "move":
                r.translate(dx, dy)
            else:
                if "l" in m: r.setLeft(r.left()   + dx)
                if "r" in m: r.setRight(r.right() + dx)
                if "t" in m: r.setTop(r.top()     + dy)
                if "b" in m: r.setBottom(r.bottom() + dy)
            # Klampoljuk a kép határaira
            if self._img_size:
                iw, ih = self._img_size
                r = r.normalized()
                r.setLeft(max(0.0, r.left()))
                r.setTop(max(0.0,  r.top()))
                r.setRight(min(float(iw),  r.right()))
                r.setBottom(min(float(ih), r.bottom()))
            self._roi_img = r.normalized()
            self.update()
            return

        # ── Hover kurzor visszajelzés ─────────────────────────────────────────
        if not self._rpan_active and not (event.buttons() & Qt.MouseButton.LeftButton):
            hit    = self._hit_test(wx, wy)
            rect   = self._display_rect()
            handle = self._hit_roi_handle(wx, wy)
            if self._poly_mode:
                self.setCursor(Qt.CursorShape.CrossCursor)
            elif handle:
                self.setCursor(self._HANDLE_CURSORS.get(
                    handle, Qt.CursorShape.ArrowCursor))
            elif self._hit_roi_body(wx, wy):
                self.setCursor(Qt.CursorShape.SizeAllCursor)
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
        if event.button() == Qt.MouseButton.LeftButton and self._roi_mode:
            self._roi_mode      = ""
            self._roi_img_start = None
            self.update()
            self.roi_rect_changed.emit(
                QRectF(self._roi_img) if self._roi_img else None)
            return

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
            # Ctrl-keret: TARTÓS téglalap-ROI létrehozása képkoordinátában
            tl = self._w2i(rx0, ry0)
            br = self._w2i(rx1, ry1)
            if tl is not None and br is not None:
                candidate = QRectF(QPointF(*tl), QPointF(*br)).normalized()
                # Minimum méret ellenőrzés: legalább 10×10 képpont kell,
                # különben apró Ctrl+húzás érvénytelen ROI-t hozna létre.
                if candidate.width() >= _ROI_MIN_SIZE and candidate.height() >= _ROI_MIN_SIZE:
                    self._roi_img = candidate
                    self.update()
                    self.roi_rect_changed.emit(QRectF(self._roi_img))
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
            # Enter → görbe lezárása ha görbe módban vagyunk
            if self._draw_mode == "polyline" and len(self._curve_pts) >= 2:
                self._finish_curve()
            elif self._poly_mode and len(self._poly_pts) >= 3:
                self._close_polygon()
            event.accept()

        elif key == Qt.Key.Key_Escape:
            # Escape → félkész görbe vagy sokszög megszakítása
            if self._draw_mode in ("polyline", "arc") and self._curve_pts:
                self._curve_pts = []
                self.update()
            elif self._poly_mode:
                self._poly_mode    = False
                self._poly_pts     = []
                self._display_poly = []
                self.update()
            event.accept()

        elif key == Qt.Key.Key_Backspace:
            # Backspace → utolsó csúcs törlése görbe módban
            if self._draw_mode in ("polyline", "arc") and self._curve_pts:
                self._curve_pts.pop()
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

    # ── Drag & Drop ──────────────────────────────────────────────────────────

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().lower().endswith(self._IMG_EXTS):
                    self._drag_hover = True
                    self.update()
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dragMoveEvent(self, event) -> None:
        event.acceptProposedAction()

    def dragLeaveEvent(self, event) -> None:
        self._drag_hover = False
        self.update()

    def dropEvent(self, event) -> None:
        self._drag_hover = False
        self.update()
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(self._IMG_EXTS):
                self.image_drop_requested.emit(path)
                event.acceptProposedAction()
                return
        event.ignore()

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
        header = QAction(tr(f"  Pont  #{index + 1}"), self)
        header.setEnabled(False)
        menu.addAction(header)
        menu.addSeparator()

        act_sel = QAction(tr("Kiválasztás"), self)
        act_sel.triggered.connect(lambda: self.point_selected.emit(index))
        menu.addAction(act_sel)

        act_del = QAction(tr("Törlés  (párjával együtt)"), self)
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

        # ── Tartós téglalap-ROI overlay ──────────────────────────────────────
        roi_w = self._roi_widget_rect()
        if roi_w is not None and rect is not None:
            # Sötétítés a ROI-n kívüli területen
            outside = QPainterPath()
            outside.addRect(QRectF(rect))
            inside = QPainterPath()
            inside.addRect(roi_w)
            shadow = outside.subtracted(inside)
            painter.setBrush(QBrush(_C_ROI_OUTSIDE))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPath(shadow)
            # ROI keret + belső fill
            painter.setPen(QPen(_C_ROI_BORDER, 2))
            painter.setBrush(QBrush(_C_ROI_FILL))
            painter.drawRect(roi_w)
            # 8 fogópont
            painter.setPen(QPen(_C_ROI_BORDER, 1.5))
            painter.setBrush(QBrush(_C_ROI_HANDLE))
            for hx, hy in self._roi_handles_w().values():
                r = _ROI_HANDLE_R
                painter.drawEllipse(int(hx) - r, int(hy) - r, r * 2, r * 2)
            # Méret-felirat
            font_roi = QFont()
            font_roi.setPointSize(9)
            painter.setFont(font_roi)
            painter.setPen(QPen(_C_ROI_BORDER, 1))
            lbl = (f"ROI  {int(self._roi_img.width())} × "
                   f"{int(self._roi_img.height())} px")
            painter.drawText(int(roi_w.left()) + 4, int(roi_w.top()) - 5, lbl)

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

        # ── Vonallánc / ív rajzolás overlay ─────────────────────────────────
        if self._draw_mode in ("polyline", "arc") and self._curve_pts:
            c_curve = QColor("#22ddcc") if self._draw_mode == "polyline" else QColor("#ffaa44")
            pen_solid = QPen(c_curve, 2, Qt.PenStyle.SolidLine)
            pen_dash  = QPen(c_curve, 1, Qt.PenStyle.DashLine)
            # Widget-koordinátákra konvertálás
            pts_w = [self._i2w(x, y) for x, y in self._curve_pts]
            pts_w = [p for p in pts_w if p is not None]
            if pts_w:
                # Összekötő élek
                painter.setPen(pen_solid)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                for k in range(len(pts_w) - 1):
                    ax, ay = pts_w[k]
                    bx, by = pts_w[k + 1]
                    painter.drawLine(int(ax), int(ay), int(bx), int(by))
                # Preview vonal az egér felé
                lx, ly = pts_w[-1]
                cx, cy = self._cursor_pos
                painter.setPen(pen_dash)
                painter.drawLine(int(lx), int(ly), int(cx), int(cy))
                # Csúcspontok (első nagyobb)
                painter.setPen(QPen(c_curve, 2))
                for k, (px, py) in enumerate(pts_w):
                    r = 7 if k == 0 else 4
                    painter.setBrush(QBrush(c_curve) if k == 0 else Qt.BrushStyle.NoBrush)
                    painter.drawEllipse(int(px) - r, int(py) - r, r * 2, r * 2)
            # Állapot / tipp szöveg
            n_pts = len(self._curve_pts)
            if self._draw_mode == "polyline":
                hint = (tr(f"Vonallánc – {n_pts} csúcs  |  "
                           "Klikk: csúcs  |  2×klikk / Enter: kész  "
                           "|  Jobb klikk / Backspace: töröl  |  Esc: mégse"))
            else:
                labels = [tr("Ív – klikk: kezdőpont"),
                          tr("Ív – klikk: közepe (az ívon)"),
                          tr("Ív – klikk: végpont  |  Jobb klikk: töröl")]
                hint = labels[min(n_pts, 2)]
            painter.setPen(QPen(c_curve, 1))
            font_c = QFont()
            font_c.setPointSize(9)
            painter.setFont(font_c)
            painter.drawText(6, self.height() - 8, hint)

        # ── Drag & Drop hover kiemelés ───────────────────────────────────────
        if self._drag_hover:
            c = QColor("#44aaff")
            painter.setPen(QPen(c, 3))
            painter.setBrush(QBrush(QColor(68, 170, 255, 25)))
            painter.drawRect(self.rect().adjusted(2, 2, -3, -3))
            font3 = QFont()
            font3.setPointSize(13)
            font3.setBold(True)
            painter.setFont(font3)
            painter.setPen(QPen(c, 1))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter,
                f"Ejtsd ide!\n({self.title})"
            )


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

    points_changed           = pyqtSignal()
    # (img_polygon, side, delete_in_roi, backend)  – sokszög-ROI keresés
    roi_search_requested     = pyqtSignal(list, str, bool, str)
    # (roi_a_QRectF, roi_b_QRectF, delete_in_roi, backend) – kétoldali ROI keresés
    dual_roi_search_requested = pyqtSignal(object, object, bool, str)
    # kép drag & drop jelzések a főablaknak
    image_a_drop_requested   = pyqtSignal(str)
    image_b_drop_requested   = pyqtSignal(str)

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
        # Görbe rajzolás állapot
        self._draw_mode:         str                      = "point"
        self._pending_curve_a:   Optional[List]           = None
        self._pending_curve_b:   Optional[List]           = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 4, 0, 0)
        root.setSpacing(4)

        # ── Eszköztár ────────────────────────────────────────────────────────
        bar = QHBoxLayout()
        bar.setSpacing(6)

        self.btn_del = QPushButton(tr("✕  Kiválasztott pár(ok) törlése"))
        self.btn_del.setEnabled(False)
        self.btn_del.setFixedHeight(28)
        self.btn_del.setStyleSheet(
            "QPushButton{background:#6b1e1e;color:#eee;padding:0 12px;"
            "border-radius:4px;font-size:12px;}"
            "QPushButton:hover{background:#922828;}"
            "QPushButton:disabled{background:#2e2e2e;color:#555;}"
        )
        self.btn_del.clicked.connect(self._delete_selection)

        self.btn_roi_search = QPushButton(tr("🔍  ROI keresés"))
        self.btn_roi_search.setEnabled(False)
        self.btn_roi_search.setFixedHeight(28)
        self.btn_roi_search.setToolTip(
            tr("Pontkeresés és párosítás csak a megjelölt ROI területeken belül\n"
            "(mindkét képen kell aktív ROI)"))
        self.btn_roi_search.setStyleSheet(
            "QPushButton{background:#1a4a2e;color:#eee;padding:0 10px;"
            "border-radius:4px;font-size:12px;}"
            "QPushButton:hover{background:#246038;}"
            "QPushButton:disabled{background:#2e2e2e;color:#555;}"
        )
        self.btn_roi_search.clicked.connect(self._on_dual_roi_search)

        self.btn_roi_clear = QPushButton(tr("⬜  ROI törlése"))
        self.btn_roi_clear.setEnabled(False)
        self.btn_roi_clear.setFixedHeight(28)
        self.btn_roi_clear.setToolTip(tr("Mindkét canvas ROI-jának törlése"))
        self.btn_roi_clear.setStyleSheet(
            "QPushButton{background:#2e2e2e;color:#aaa;padding:0 10px;"
            "border-radius:4px;font-size:12px;}"
            "QPushButton:hover{background:#3a3a3a;}"
            "QPushButton:disabled{background:#2e2e2e;color:#555;}"
        )
        self.btn_roi_clear.clicked.connect(self._clear_rois)

        self.lbl_hint = QLabel(
            "2× klikk: pont  |  Húzás: kijelölés  |  "
            "Ctrl+húzás: ROI keret  |  ROI-n: húzás=mozgat, sarok=átméretez  |  "
            "Ctrl+klikk: sokszög-ROI  |  Delete: törlés  |  Görgetés: zoom"
        )
        self.lbl_hint.setStyleSheet("color:#666;font-size:11px;")

        self.lbl_count = QLabel(tr("Pontpárok: 0"))
        self.lbl_count.setStyleSheet("color:#aaa;font-size:12px;")

        bar.addWidget(self.btn_del)
        bar.addSpacing(4)
        bar.addWidget(self.btn_roi_search)
        bar.addWidget(self.btn_roi_clear)
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

        # ── Görbe rajzolás eszköztár ─────────────────────────────────────────
        cbar = QHBoxLayout()
        cbar.setSpacing(6)

        _MODE_BASE = ("QPushButton{padding:0 10px;border-radius:4px;font-size:12px;"
                      "border:1px solid #444;}"
                      "QPushButton:checked{border:1px solid #22ddcc;}")
        _btn_style = lambda col: (
            f"QPushButton{{background:#1e2430;color:#bbb;padding:0 10px;"
            f"border-radius:4px;font-size:12px;border:1px solid #444;}}"
            f"QPushButton:checked{{background:{col};color:#111;border:1px solid {col};}}"
            f"QPushButton:hover{{background:#2a3040;}}"
        )

        self.btn_mode_point    = QPushButton(tr("• Pont"))
        self.btn_mode_polyline = QPushButton(tr("— Vonallánc"))
        self.btn_mode_arc      = QPushButton(tr("⌒ Ív"))
        for btn, col in [(self.btn_mode_point, "#aaa"),
                         (self.btn_mode_polyline, "#22ddcc"),
                         (self.btn_mode_arc, "#ffaa44")]:
            btn.setCheckable(True)
            btn.setFixedHeight(26)
            btn.setStyleSheet(_btn_style(col))
        self.btn_mode_point.setChecked(True)
        self.btn_mode_point.setToolTip(
            tr("Normál mód: dupla kattintással adj hozzá pontpárokat"))
        self.btn_mode_polyline.setToolTip(
            tr("Vonallánc: rajzolj töröttvonalat mindkét képre → "
               "automatikusan felosztja pontpárokra"))
        self.btn_mode_arc.setToolTip(
            tr("Ív: 3 kattintással definiálj körívvet (kezdő, közepe, vég) → "
               "felosztja pontpárokra"))

        self.btn_mode_point.clicked.connect(
            lambda: self._set_draw_mode("point"))
        self.btn_mode_polyline.clicked.connect(
            lambda: self._set_draw_mode("polyline"))
        self.btn_mode_arc.clicked.connect(
            lambda: self._set_draw_mode("arc"))

        lbl_n = QLabel(tr("Pontok/görbe:"))
        lbl_n.setStyleSheet("color:#888;font-size:12px;")
        self.spin_subdiv = QSpinBox()
        self.spin_subdiv.setRange(3, 60)
        self.spin_subdiv.setValue(10)
        self.spin_subdiv.setFixedWidth(52)
        self.spin_subdiv.setFixedHeight(26)
        self.spin_subdiv.setToolTip(
            tr("Hány pontpárt hozzon létre egy vonallánc/ív szakaszból"))
        self.spin_subdiv.setStyleSheet(
            "QSpinBox{background:#1e2430;color:#ddd;border:1px solid #444;"
            "border-radius:4px;padding:0 4px;}"
            "QSpinBox::up-button,QSpinBox::down-button{width:14px;}")

        self.lbl_curve_status = QLabel("")
        self.lbl_curve_status.setStyleSheet(
            "color:#22ddcc;font-size:12px;font-weight:bold;")

        cbar.addWidget(self.btn_mode_point)
        cbar.addWidget(self.btn_mode_polyline)
        cbar.addWidget(self.btn_mode_arc)
        cbar.addSpacing(10)
        cbar.addWidget(lbl_n)
        cbar.addWidget(self.spin_subdiv)
        cbar.addSpacing(10)
        cbar.addWidget(self.lbl_curve_status)
        cbar.addStretch()

        root.addLayout(bar)
        root.addLayout(cbar)
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
        # Tartós ROI szinkronizálás
        self.canvas_a.roi_rect_changed.connect(self._on_roi_a_changed)
        self.canvas_b.roi_rect_changed.connect(self._on_roi_b_changed)
        # Kép drag & drop → tovább a főablaknak
        self.canvas_a.image_drop_requested.connect(self.image_a_drop_requested)
        self.canvas_b.image_drop_requested.connect(self.image_b_drop_requested)
        # Zoom szinkron
        self.canvas_a.zoom_sync_requested.connect(self.canvas_b.set_zoom_view)
        self.canvas_b.zoom_sync_requested.connect(self.canvas_a.set_zoom_view)
        # Középső gomb + Ctrl+Z
        self.canvas_a.middle_clicked.connect(self._on_middle_click)
        self.canvas_b.middle_clicked.connect(self._on_middle_click)
        self.canvas_a.undo_requested.connect(self._pop_undo)
        self.canvas_b.undo_requested.connect(self._pop_undo)
        # Görbe kész jelek
        self.canvas_a.curve_done.connect(self._on_curve_done_a)
        self.canvas_b.curve_done.connect(self._on_curve_done_b)

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
        self.lbl_count.setText(tr(f"Pontpárok: {n}"))

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

    # ── Görbe rajzolás logika ────────────────────────────────────────────────

    def _set_draw_mode(self, mode: str) -> None:
        """Rajzolási mód váltása: pont / vonallánc / ív."""
        self._draw_mode      = mode
        self._pending_curve_a = None
        self._pending_curve_b = None
        self.canvas_a.set_draw_mode(mode)
        self.canvas_b.set_draw_mode(mode)
        # Gomb állapotok
        self.btn_mode_point.setChecked(mode == "point")
        self.btn_mode_polyline.setChecked(mode == "polyline")
        self.btn_mode_arc.setChecked(mode == "arc")
        self._update_curve_status()

    def _update_curve_status(self) -> None:
        """Állapotsor frissítése görbe módban."""
        if self._draw_mode == "point":
            self.lbl_curve_status.setText("")
            return
        if self._pending_curve_a is None:
            n = "ív" if self._draw_mode == "arc" else "vonalat"
            self.lbl_curve_status.setStyleSheet(
                "color:#22ddcc;font-size:12px;font-weight:bold;")
            self.lbl_curve_status.setText(
                tr(f"← Rajzolj {n} az  A  képre"))
        elif self._pending_curve_b is None:
            n = "ív" if self._draw_mode == "arc" else "vonalat"
            self.lbl_curve_status.setStyleSheet(
                "color:#ffaa44;font-size:12px;font-weight:bold;")
            self.lbl_curve_status.setText(
                tr(f"✔ A kész  ·  Most rajzolj {n} a  B  képre"))
        else:
            self.lbl_curve_status.setText("")

    def _on_curve_done_a(self, pts: list) -> None:
        self._pending_curve_a = pts
        # B oldalt is frissítjük a várakozási jelzéssel
        self.canvas_b.set_draw_mode(self._draw_mode)
        self._update_curve_status()
        if self._pending_curve_b is not None:
            self._commit_curves()

    def _on_curve_done_b(self, pts: list) -> None:
        self._pending_curve_b = pts
        self._update_curve_status()
        if self._pending_curve_a is not None:
            self._commit_curves()

    def _commit_curves(self) -> None:
        """Mindkét görbe kész: felosztja és hozzáadja pontpárként."""
        n   = self.spin_subdiv.value()
        pa  = self._pending_curve_a
        pb  = self._pending_curve_b
        self._pending_curve_a = None
        self._pending_curve_b = None
        self._update_curve_status()

        mode = self._draw_mode
        if mode == "polyline":
            pts_a = self._subdivide_polyline(pa, n)
            pts_b = self._subdivide_polyline(pb, n)
        elif mode == "arc" and len(pa) == 3 and len(pb) == 3:
            pts_a = self._subdivide_arc(*pa, n)
            pts_b = self._subdivide_arc(*pb, n)
        else:
            return

        if not pts_a or not pts_b:
            return

        self._push_undo()
        for (ax, ay), (bx, by) in zip(pts_a, pts_b):
            self.project.anchor_points_a.append([ax, ay])
            self.project.anchor_points_b.append([bx, by])
        self._sync()
        self.points_changed.emit()

    # ── Görbe felosztó segédek ───────────────────────────────────────────────

    @staticmethod
    def _subdivide_polyline(pts: list, n: int) -> list:
        """N egyenközű pontot mintavételez egy töröttvonal mentén (képkoord)."""
        if n <= 0 or len(pts) < 2:
            return list(pts)
        if n == 1:
            return [pts[0]]
        # Kumulatív ívhosszak
        lengths = [0.0]
        for i in range(1, len(pts)):
            dx = pts[i][0] - pts[i - 1][0]
            dy = pts[i][1] - pts[i - 1][1]
            lengths.append(lengths[-1] + math.hypot(dx, dy))
        total = lengths[-1]
        if total < 1e-9:
            return [tuple(pts[0])] * n
        result = []
        for k in range(n):
            t = total * k / (n - 1)
            for seg in range(1, len(lengths)):
                if lengths[seg] >= t - 1e-9:
                    t0, t1 = lengths[seg - 1], lengths[seg]
                    if t1 - t0 < 1e-9:
                        result.append(tuple(pts[seg - 1]))
                    else:
                        alpha = (t - t0) / (t1 - t0)
                        x = pts[seg-1][0] + alpha * (pts[seg][0] - pts[seg-1][0])
                        y = pts[seg-1][1] + alpha * (pts[seg][1] - pts[seg-1][1])
                        result.append((x, y))
                    break
            else:
                result.append(tuple(pts[-1]))
        return result

    @staticmethod
    def _subdivide_arc(p0, p1, p2, n: int) -> list:
        """N egyenközű pontot mintavételez a P0-P1-P2 körívven (P1 az ívon van)."""
        if n <= 0:
            return []
        if n == 1:
            return [tuple(p0)]
        ax, ay = p0;  bx, by = p1;  cx, cy = p2
        D = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(D) < 1e-10:
            # Kollineáris → egyenes vonal
            return PointEditorWidget._subdivide_polyline([p0, p2], n)
        ux = ((ax**2+ay**2)*(by-cy) + (bx**2+by**2)*(cy-ay) + (cx**2+cy**2)*(ay-by)) / D
        uy = ((ax**2+ay**2)*(cx-bx) + (bx**2+by**2)*(ax-cx) + (cx**2+cy**2)*(bx-ax)) / D
        r  = math.hypot(ax - ux, ay - uy)
        a_start = math.atan2(ay - uy, ax - ux)
        a_mid   = math.atan2(by - uy, bx - ux)
        a_end   = math.atan2(cy - uy, cx - ux)
        # Iránymeghatározás: az ív P1-en megy keresztül
        m_ccw = (a_mid - a_start) % (2 * math.pi)
        e_ccw = (a_end - a_start) % (2 * math.pi)
        if e_ccw == 0:
            e_ccw = 2 * math.pi
        if m_ccw <= e_ccw:          # CCW
            angles = [a_start + e_ccw * k / (n - 1) for k in range(n)]
        else:                        # CW
            sweep = (2 * math.pi - e_ccw) % (2 * math.pi) or 2 * math.pi
            angles = [a_start - sweep * k / (n - 1) for k in range(n)]
        return [(ux + r * math.cos(a), uy + r * math.sin(a)) for a in angles]

    # ── Tartós téglalap-ROI kezelés ──────────────────────────────────────────

    def _on_roi_a_changed(self, roi_img) -> None:
        """A kép ROI-ja változott → arányos ROI megjelenítése B képen."""
        self._update_roi_buttons()
        if roi_img is None:
            return
        img_a = self.project.image_a
        img_b = self.project.image_b
        if img_a is None or img_b is None:
            return
        ha, wa = img_a.shape[:2]
        hb, wb = img_b.shape[:2]
        # Arányos átméretezés A → B képkoord-rendszerbe
        ax = roi_img.left()  / wa;  ay = roi_img.top()    / ha
        aw = roi_img.width() / wa;  ah = roi_img.height() / ha
        b_rect = QRectF(ax * wb, ay * hb, aw * wb, ah * hb)
        # Canvas B szinkronizálása (szignál kikapcsolva körkörös hívás ellen)
        self.canvas_b.roi_rect_changed.disconnect(self._on_roi_b_changed)
        self.canvas_b.set_roi_from_image(b_rect)
        self.canvas_b.roi_rect_changed.connect(self._on_roi_b_changed)
        self._update_roi_buttons()

    def _on_roi_b_changed(self, roi_img) -> None:
        """B kép ROI-ja változott (felhasználó módosította)."""
        self._update_roi_buttons()

    def _update_roi_buttons(self) -> None:
        has_a = self.canvas_a.get_roi_image() is not None
        has_b = self.canvas_b.get_roi_image() is not None
        self.btn_roi_search.setEnabled(has_a and has_b)
        self.btn_roi_clear.setEnabled(has_a or has_b)

    def _clear_rois(self) -> None:
        self.canvas_a.clear_roi()
        self.canvas_b.clear_roi()
        self._update_roi_buttons()

    def _on_dual_roi_search(self) -> None:
        """ROI keresés indítása – backend választó menüvel."""
        roi_a = self.canvas_a.get_roi_image()
        roi_b = self.canvas_b.get_roi_image()
        if roi_a is None or roi_b is None:
            return

        # Backend választó + delete_in_roi – ugyanolyan menüstílussal
        menu = QMenu(self)
        menu.setStyleSheet(self._MENU_STYLE)

        sub = menu.addMenu("Backend")
        ag  = QActionGroup(sub)
        ag.setExclusive(True)
        for be in self._BACKENDS:
            act = QAction(be, sub, checkable=True)
            act.setChecked(be == self._roi_last_backend)
            ag.addAction(act)
            sub.addAction(act)

        act_del = QAction(tr("ROI-n belüli meglévő párokat törölje előbb"), menu,
                          checkable=True)
        act_del.setChecked(self._roi_delete_in_roi)
        menu.addAction(act_del)
        menu.addSeparator()
        act_go  = QAction(tr("🔍  Keresés indítása"), menu)
        menu.addAction(act_go)

        chosen = menu.exec(self.btn_roi_search.mapToGlobal(
            self.btn_roi_search.rect().bottomLeft()))
        if chosen is None or chosen is not act_go:
            return

        # Frissítjük az állapotot
        for act in ag.actions():
            if act.isChecked():
                self._roi_last_backend = act.text()
                break
        self._roi_delete_in_roi = act_del.isChecked()

        self.dual_roi_search_requested.emit(
            roi_a, roi_b,
            self._roi_delete_in_roi,
            self._roi_last_backend,
        )

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
        title_act  = QAction(tr(f"  {shape_name}  –  keresési beállítások"), menu)
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
        del_act = QAction(tr("Meglévő pontok törlése a területen belül"), menu)
        del_act.setCheckable(True)
        del_act.setChecked(self._roi_delete_in_roi)
        del_act.setToolTip(
            tr("Ha be van jelölve, keresés előtt törli az összes meglévő\n"
            "pontpárt amelyek bármelyik tagja a berajzolt területen belül esik."))
        menu.addAction(del_act)
        menu.addSeparator()

        # Keresés gomb
        search_act = QAction(tr(f"▶  ROI keresés indítása"), menu)
        menu.addAction(search_act)

        cancel_act = QAction(tr("✕  Mégse"), menu)
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
