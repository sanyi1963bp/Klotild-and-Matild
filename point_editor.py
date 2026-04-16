"""
point_editor.py  –  Pontszerkesztő modul (ArchMorph Professional)
==================================================================
Önálló PyQt6 modul; csak numpy és opcionálisan cv2 kell hozzá.

Interakció
----------
Egyszeres klikk         → pontpár lerakása (3-pont affin interpoláció a párhoz)
Dupla klikk             → vonallánc rajzolás kezdete / vége
  (közben egyszeres klikk → csúcs hozzáadása; párja azonnal megjelenik)
  Vonallánc lezárása után → felosztás 30 px-enként, interpolált párpontok
Bal húzás ponton        → pont mozgatása
Bal húzás képen         → gumiszalag-kijelölés (kék keret)
Ctrl+bal húzás          → tartós téglalap-ROI
Ctrl+bal klikk ponton  → multi-kijelölés toggle
Jobb klikk képen        → pásztázás; klikk (mozgás nélkül) → pont törlés
Középső gomb            → zoom visszaállítás + utolsó undo
Ctrl+görgetés           → pontindex léptetés
Görgetés                → zoom
Delete                  → kijelölt pontpár(ok) törlése
Ctrl+Z                  → visszavonás
Nyílbillentyűk          → aktív pont mozgatása 1 px (Shift = 10 px)
Space                   → zoom szinkron a másik canvasra
Ctrl+0                  → zoom visszaállítás
Esc                     → vonallánc rajzolás megszakítása

Exportált osztályok
-------------------
    PointEditorCanvas   – egy képet megjelenítő szerkeszthető canvas
    PointEditorWidget   – kétpaneles szerkesztő (A + B kép, párkezelés)
"""
from __future__ import annotations

import math
from typing import List, Optional, Set, Tuple

import numpy as np

from PyQt6.QtCore import Qt, QPointF, QRectF, QPoint, QSize, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QAction, QActionGroup, QBrush, QColor, QFont,
    QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF,
)
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QMenu, QPushButton, QMessageBox,
    QSizePolicy, QSplitter, QVBoxLayout, QWidget,
)

try:
    from TRANSLATIONS import tr
except ImportError:
    def tr(text: str) -> str: return text

# ── Konfiguráció betöltése ────────────────────────────────────────────────────
try:
    from archmorph_config_loader import cfg, cfg_rgba
except ImportError:
    def cfg(key, default):      return default        # type: ignore[misc]
    def cfg_rgba(key, default): return default        # type: ignore[misc]

# ── Vizuális konstansok ───────────────────────────────────────────────────────

_R_NORMAL = cfg("ui.sizes.point_radius_normal", 6)
_R_ACTIVE = cfg("ui.sizes.point_radius_active", 10)
_R_MULTI  = cfg("ui.sizes.point_radius_multi",  8)
_HIT_R    = cfg("ui.sizes.point_hit_radius",    14)

_C_NORMAL = QColor(cfg("ui.colors.points.normal",      "#ff8a3d"))
_C_ACTIVE = QColor(cfg("ui.colors.points.active",      "#ff3333"))
_C_MULTI  = QColor(cfg("ui.colors.points.multi",       "#44aaff"))
_C_TEXT   = QColor(cfg("ui.colors.points.text",        "#ffffff"))
_C_TEXT_A = QColor(cfg("ui.colors.points.text_active", "#ffcccc"))
_C_BG     = QColor(cfg("ui.colors.canvas.background",  "#1a1f24"))
_C_BORDER = QColor(cfg("ui.colors.canvas.border",      "#343b45"))
_C_PLACEHOLDER = QColor(cfg("ui.colors.canvas.placeholder", "#4a5260"))

_C_FILL_N = QColor(*cfg_rgba("ui.colors.points.fill.normal_rgba", (255, 138,  61,  70)))
_C_FILL_A = QColor(*cfg_rgba("ui.colors.points.fill.active_rgba", (255,  51,  51, 150)))
_C_FILL_M = QColor(*cfg_rgba("ui.colors.points.fill.multi_rgba",  ( 68, 170, 255,  90)))

_C_RBAND_SEL = QColor(cfg("ui.colors.rband.selection", "#88bbff"))
_C_RBAND_ROI = QColor(cfg("ui.colors.rband.roi",       "#ffcc44"))

_C_POLY_DRAW = QColor(cfg("ui.colors.polygon.draw", "#FFB300"))
_C_POLY_DONE = QColor(cfg("ui.colors.polygon.done", "#FF6F00"))
_C_POLY_FILL = QColor(*cfg_rgba("ui.colors.polygon.fill_rgba", (255, 179, 0, 35)))

_C_ROI_BORDER  = QColor(cfg("ui.colors.roi.border", "#ff9900"))
_C_ROI_HANDLE  = QColor(cfg("ui.colors.roi.handle", "#ffffff"))
_C_ROI_FILL    = QColor(*cfg_rgba("ui.colors.roi.fill_rgba",    (255, 153,   0,  22)))
_C_ROI_OUTSIDE = QColor(*cfg_rgba("ui.colors.roi.outside_rgba", (  0,   0,   0,  90)))
_ROI_HANDLE_R  = cfg("ui.sizes.roi_handle_radius", 6)
_ROI_MIN_SIZE  = cfg("ui.sizes.roi_min_size",       10)

# Vonallánc szín
_C_POLYLINE = QColor("#22ddcc")

# Klikk-disambiguáció időzítője (ms) – egyszeres vs. dupla klikk megkülönböztetése
_CLICK_DISAMBIG_MS = 250


# ── Segédfüggvény ─────────────────────────────────────────────────────────────
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

    Jelek
    -----
    single_click_confirmed(x, y)     250 ms utáni egyszeres klikk (képkoord)
    double_click_confirmed(x, y)     dupla klikk (képkoord)
    escape_pressed()                 Esc lenyomva
    point_moved(idx, x, y)           pont pozíciója megváltozott
    point_deleted(idx)               pont törlése kérve
    points_delete_multi_requested()  multi-kijelölt pontok törlése kérve
    point_selected(idx)              pont kiválasztva (-1 = deselect)
    selection_changed(list[int])     gumiszalag kijelölés végeredménye
    roi_ready(list)                  sokszög-ROI kész
    roi_rect_changed(QRectF|None)    tartós téglalap-ROI változott
    zoom_sync_requested(z, rx, ry)   Space: zoom-szinkron kérés
    middle_clicked()                 középső gomb
    undo_requested()                 Ctrl+Z
    image_drop_requested(str)        kép drag & drop
    """

    single_click_confirmed       = pyqtSignal(float, float)
    double_click_confirmed       = pyqtSignal(float, float)
    escape_pressed               = pyqtSignal()
    point_moved                  = pyqtSignal(int, float, float)
    point_deleted                = pyqtSignal(int)
    points_delete_multi_requested = pyqtSignal()
    point_selected               = pyqtSignal(int)
    selection_changed            = pyqtSignal(list)
    roi_ready                    = pyqtSignal(list)
    roi_rect_changed             = pyqtSignal(object)
    zoom_sync_requested          = pyqtSignal(float, float, float)
    middle_clicked               = pyqtSignal()
    undo_requested               = pyqtSignal()
    image_drop_requested         = pyqtSignal(str)
    polyline_vertex_moved        = pyqtSignal(int, int, float, float)  # pidx, vidx, x, y
    polyline_delete_requested    = pyqtSignal(int)                      # pidx
    polyline_vertex_delete_requested = pyqtSignal(int, int)             # pidx, vidx
    # Keresési maszk jelzések
    smask_draw_finished          = pyqtSignal(list)   # norm. koordináták, sokszög lezárásakor
    smask_poly_changed           = pyqtSignal(list)   # minden változáskor (gomb enable/disable)

    _IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    _ZOOM_MIN  = 0.5
    _ZOOM_MAX  = 20.0
    _ZOOM_STEP = 1.18
    _RBAND_MIN = 4

    def __init__(self, title: str) -> None:
        super().__init__()
        self.title = title

        self._pixmap_orig: Optional[QPixmap]           = None
        self._img_size:    Optional[Tuple[int, int]]   = None
        self._points:      List[Tuple[float, float]]   = []
        self._selected:    int                         = -1
        self._selected_set: Set[int]                   = set()

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
        self._rband_p0:     Optional[Tuple[float, float]] = None
        self._rband_p1:     Optional[Tuple[float, float]] = None
        self._rband_active: bool = False
        self._rband_ctrl:   bool = False

        # Sokszög-ROI rajzolás
        self._poly_pts:    List[Tuple[float, float]] = []
        self._poly_mode:   bool                      = False
        self._cursor_pos:  Tuple[float, float]       = (0.0, 0.0)
        self._display_poly: List[Tuple[float, float]]= []

        # Tartós téglalap-ROI
        self._roi_img:       Optional[QRectF]  = None
        self._roi_mode:      str               = ""
        self._roi_drag_wx:   float             = 0.0
        self._roi_drag_wy:   float             = 0.0
        self._roi_img_start: Optional[QRectF]  = None

        # Drag & drop
        self._drag_hover: bool = False
        self.setAcceptDrops(True)

        # Zoom horgony
        self._zoom_anchor: Optional[Tuple[float, float]] = None

        # ── Klikk-disambiguáció (250 ms timer) ──────────────────────────────
        self._click_timer = QTimer(self)
        self._click_timer.setSingleShot(True)
        self._click_timer.setInterval(_CLICK_DISAMBIG_MS)
        self._click_timer.timeout.connect(self._on_click_timer)
        self._pending_click_pos: Optional[Tuple[float, float]] = None  # képkoord

        # ── Tárolt vonalak (a Widget állítja) ────────────────────────────────
        self._polylines_data: List[List[Tuple[float, float]]] = []
        # Vonallánc csúcs-húzás (500 ms hosszú nyomás engedélyezi)
        self._poly_drag_pidx:  int  = -1
        self._poly_drag_vidx:  int  = -1
        self._poly_dragging:   bool = False
        self._poly_drag_ready: bool = False   # True: 500 ms eltelt, húzás engedélyezve
        # Hosszú-nyomás timer (500 ms)
        self._poly_longpress_timer = QTimer(self)
        self._poly_longpress_timer.setSingleShot(True)
        self._poly_longpress_timer.setInterval(500)
        self._poly_longpress_timer.timeout.connect(self._on_poly_longpress)
        # Hover kiemelés
        self._poly_hover_pidx: int = -1
        self._poly_hover_vidx: int = -1

        # ── Keresési maszk sokszög ───────────────────────────────────────────
        self._smask_poly:        List[Tuple[float, float]]   = []    # norm. (0-1) képkoord
        self._smask_drawing:     bool                        = False  # rajzolás folyamatban
        self._smask_draw_cursor: Optional[Tuple[float, float]] = None # egér poz. rajzolás közben
        self._smask_drag_idx:    int                         = -1     # húzott csúcs indexe
        self._smask_mode:        bool                        = False  # maszk mód aktív

        # ── Vonallánc előnézet (a Widget állítja be) ─────────────────────────
        self._preview_polyline:    Optional[List[Tuple[float, float]]] = None
        self._preview_cursor_line: bool = False   # True: szaggatott vonal az egérig

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
        self._selected_set = set(indices)
        self.update()

    def set_preview_polyline(self,
                              pts: Optional[List[Tuple[float, float]]],
                              show_cursor_line: bool = False) -> None:
        """
        Vonallánc előnézet megjelenítése (a PointEditorWidget hívja).
        pts=None esetén töröl.
        show_cursor_line=True: szaggatott vonal az utolsó ponttól az egérig.
        """
        self._preview_polyline    = list(pts) if pts is not None else None
        self._preview_cursor_line = show_cursor_line
        self.update()

    def set_polylines(self, data: list) -> None:
        """
        Tárolt vonalak beállítása.
        data: [[pt0, pt1, ...], ...]  – minden elem egy vonallánc képkoord-listája.
        """
        self._polylines_data = [list(pl) for pl in data]
        self.update()

    # ── Keresési maszk API ───────────────────────────────────────────────────

    def set_smask_mode(self, active: bool) -> None:
        """Maszk-rajzolás mód be/ki. Kikapcsoláskor a rajzolási kurzor törlődik."""
        self._smask_mode = active
        if not active:
            self._smask_drawing     = False
            self._smask_draw_cursor = None
            self._smask_drag_idx    = -1
        self.update()

    def set_smask_poly(self, pts: List[Tuple[float, float]]) -> None:
        """Maszk sokszög beállítása kívülről (tükrözéshez). Norm. (0-1) koordináták."""
        self._smask_poly     = list(pts)
        self._smask_drawing  = False
        self._smask_drag_idx = -1
        self.smask_poly_changed.emit(list(self._smask_poly))
        self.update()

    def clear_smask(self) -> None:
        """Maszk törlése."""
        self._smask_poly        = []
        self._smask_drawing     = False
        self._smask_draw_cursor = None
        self._smask_drag_idx    = -1
        self.smask_poly_changed.emit([])
        self.update()

    def get_smask_pixels(self) -> Optional[np.ndarray]:
        """Maszk sokszög pixel-koordinátákban (N×2 float32 array), vagy None."""
        if len(self._smask_poly) < 3 or self._img_size is None:
            return None
        iw, ih = self._img_size
        return np.array([(nx * iw, ny * ih) for nx, ny in self._smask_poly],
                        dtype=np.float32)

    def _smask_hit(self, wx: float, wy: float) -> int:
        """Melyik maszk-csúcs van a hit-radiuszon belül? -1 ha egyik sem."""
        if not self._smask_poly or self._img_size is None:
            return -1
        iw, ih = self._img_size
        for i, (nx, ny) in enumerate(self._smask_poly):
            wpt = self._i2w(nx * iw, ny * ih)
            if wpt and math.hypot(wx - wpt[0], wy - wpt[1]) <= _HIT_R:
                return i
        return -1

    def _smask_close_drawing(self) -> None:
        """Befejezi a maszk-rajzolást és jelzést küld."""
        if len(self._smask_poly) >= 3:
            self._smask_drawing     = False
            self._smask_draw_cursor = None
            self.smask_draw_finished.emit(list(self._smask_poly))
            self.smask_poly_changed.emit(list(self._smask_poly))
            self.update()

    def _hit_test_poly_vertex(self, wx: float, wy: float) -> Tuple[int, int]:
        """
        Vonallánc csúcsát teszteli hit-radiuszon belül.
        Visszatér: (polyline_index, vertex_index) vagy (-1, -1).
        """
        for pidx, poly in enumerate(self._polylines_data):
            for vidx, pt in enumerate(poly):
                wpt = self._i2w(float(pt[0]), float(pt[1]))
                if wpt is None:
                    continue
                if math.hypot(wx - wpt[0], wy - wpt[1]) <= _HIT_R:
                    return pidx, vidx
        return -1, -1

    # ── Tartós téglalap-ROI – publikus API ──────────────────────────────────

    def set_roi_from_image(self, rect: Optional[QRectF]) -> None:
        self._roi_img = QRectF(rect) if rect is not None else None
        self.update()

    def get_roi_image(self) -> Optional[QRectF]:
        return QRectF(self._roi_img) if self._roi_img is not None else None

    def clear_roi(self) -> None:
        self._roi_img   = None
        self._roi_mode  = ""
        self.update()
        self.roi_rect_changed.emit(None)

    # ── Tartós téglalap-ROI – belső segédek ─────────────────────────────────

    def _roi_widget_rect(self) -> Optional[QRectF]:
        if self._roi_img is None:
            return None
        tl = self._i2w(self._roi_img.left(),  self._roi_img.top())
        br = self._i2w(self._roi_img.right(), self._roi_img.bottom())
        if tl is None or br is None:
            return None
        return QRectF(QPointF(*tl), QPointF(*br)).normalized()

    _HANDLE_NAMES = ("tl", "tc", "tr", "ml", "mr", "bl", "bc", "br")

    def _roi_handles_w(self) -> dict:
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
        for name, (hx, hy) in self._roi_handles_w().items():
            if math.hypot(wx - hx, wy - hy) <= _ROI_HANDLE_R + 4:
                return name
        return ""

    def _hit_roi_body(self, wx: float, wy: float) -> bool:
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

    # ── Klikk-timer ─────────────────────────────────────────────────────────

    def _on_click_timer(self) -> None:
        """250 ms elteltével: egyszeres klikk megerősítése."""
        if self._pending_click_pos is not None:
            self.single_click_confirmed.emit(*self._pending_click_pos)
            self._pending_click_pos = None

    def _on_poly_longpress(self) -> None:
        """500 ms elteltével: vonallánc csúcs mozgatás engedélyezése."""
        if self._poly_drag_pidx >= 0:
            self._poly_drag_ready = True
            self.update()   # vizuális visszajelzés: csúcs kinézete megváltozik

    # ── Egéresemények ────────────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        wx  = float(event.position().x())
        wy  = float(event.position().y())
        btn = event.button()

        # ── Maszk mód ────────────────────────────────────────────────────────
        if self._smask_mode:
            if btn == Qt.MouseButton.LeftButton and self._img_size is not None:
                ic = self._w2i(wx, wy)
                if ic is not None:
                    iw, ih = self._img_size
                    nx, ny = ic[0] / iw, ic[1] / ih
                    if self._smask_drawing and len(self._smask_poly) >= 3:
                        # Közel van az első csúcshoz → lezárás
                        fx, fy = self._smask_poly[0]
                        wpt0   = self._i2w(fx * iw, fy * ih)
                        if wpt0 and math.hypot(wx - wpt0[0], wy - wpt0[1]) <= 12:
                            self._smask_close_drawing()
                            return
                    if self._smask_drawing:
                        # Új csúcs hozzáadása
                        self._smask_poly.append((nx, ny))
                        self.update()
                    else:
                        # Meglévő csúcs fogása
                        hit = self._smask_hit(wx, wy)
                        if hit >= 0:
                            self._smask_drag_idx = hit
                        else:
                            # Új sokszög rajzolás kezdése
                            self._smask_poly    = [(nx, ny)]
                            self._smask_drawing = True
                            self.update()
            return

        # ── Középső gomb ─────────────────────────────────────────────────────
        if btn == Qt.MouseButton.MiddleButton:
            self._reset_zoom()
            self.middle_clicked.emit()
            return

        # ── Jobb gomb – pásztázás ────────────────────────────────────────────
        if btn == Qt.MouseButton.RightButton:
            self._rpan_active = True
            self._rpan_moved  = False
            self._rpan_lx     = wx
            self._rpan_ly     = wy
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        # ── Bal gomb ─────────────────────────────────────────────────────────
        if btn == Qt.MouseButton.LeftButton:
            ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)

            # ── ROI fogópont / mozgatás ──────────────────────────────────────
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

            # ── Vonallánc csúcs fogása (500 ms hosszú nyomás engedélyezi) ────
            if not ctrl:
                ppidx, pvidx = self._hit_test_poly_vertex(wx, wy)
                if ppidx >= 0:
                    self._poly_drag_pidx   = ppidx
                    self._poly_drag_vidx   = pvidx
                    self._poly_dragging    = False
                    self._poly_drag_ready  = False
                    # Klikk-timer leállítása: ne induljon pontlerakás
                    self._click_timer.stop()
                    self._pending_click_pos = None
                    # Hosszú-nyomás timer indítása
                    self._poly_longpress_timer.start(500)
                    return

            idx = self._hit_test(wx, wy)
            if idx >= 0:
                if ctrl:
                    # Ctrl + klikk ponton → multi-kijelölés toggle
                    if idx in self._selected_set:
                        self._selected_set.discard(idx)
                    else:
                        self._selected_set.add(idx)
                    self.selection_changed.emit(list(self._selected_set))
                else:
                    # Bal klikk ponton → kijelölés + húzás előkészítése
                    self._drag_idx = idx
                    self._dragging = False
                    if idx not in self._selected_set:
                        self._selected_set.clear()
                    self.point_selected.emit(idx)
            else:
                # Klikk üres területen → gumiszalag VAGY pontlerakás (timer)
                coords = self._w2i(wx, wy)
                if coords is not None:
                    self._rband_p0    = (wx, wy)
                    self._rband_p1    = (wx, wy)
                    self._rband_ctrl  = ctrl
                    self._rband_active = False
                    if not ctrl:
                        # Soft-zone: ha közel vagyunk egy vonallánc-csúcshoz
                        # (2× hit radius), NE indítsuk a pontlerakó timert –
                        # a klikk valószínűleg eltévesztett csúcsfogás kísérlet.
                        near_vertex = False
                        for _poly in self._polylines_data:
                            for _pt in _poly:
                                _wpt = self._i2w(float(_pt[0]), float(_pt[1]))
                                if (_wpt is not None and
                                        math.hypot(wx - _wpt[0],
                                                   wy - _wpt[1]) <= _HIT_R * 2):
                                    near_vertex = True
                                    break
                            if near_vertex:
                                break
                        if not near_vertex:
                            # Elindítjuk a 250 ms timert az egyszeres klikk detektálásához
                            self._pending_click_pos = coords
                            self._click_timer.start(_CLICK_DISAMBIG_MS)

        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        """
        Dupla klikk: klikk-timer törlése + double_click_confirmed jel.
        Ctrl+dupla-klikk sokszög módban: sokszög lezárása.
        Vonallánc csúcson: elnyelés (ne indítson új vonalat).
        """
        if event.button() == Qt.MouseButton.LeftButton:
            # Maszk mód: dupla klikk = sokszög lezárása
            if self._smask_mode:
                self._smask_close_drawing()
                return

            wx   = float(event.position().x())
            wy   = float(event.position().y())
            ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)

            # Timer törlése: ez NEM egyszeres klikk
            self._click_timer.stop()
            self._pending_click_pos = None
            self._poly_longpress_timer.stop()
            self._poly_drag_ready = False
            self._poly_drag_pidx  = -1
            self._poly_drag_vidx  = -1

            # Dupla klikk vonallánc csúcson → ne indítson új vonalat
            if not ctrl:
                ppidx, _ = self._hit_test_poly_vertex(wx, wy)
                if ppidx >= 0:
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

            if self._poly_mode:
                # Dupla klikk Ctrl nélkül sokszög módban → megszakítás
                self._poly_mode = False
                self._poly_pts  = []
                self._rband_active = False
                self._rband_p0     = None
                self._rband_p1     = None
                self.update()
                super().mouseDoubleClickEvent(event)
                return

            # Normál dupla klikk: double_click_confirmed jel
            self._rband_active = False
            self._rband_p0     = None
            self._rband_p1     = None
            coords = self._w2i(wx, wy)
            if coords is not None and not ctrl:
                self.double_click_confirmed.emit(*coords)

        super().mouseDoubleClickEvent(event)

    def _close_polygon(self) -> None:
        if len(self._poly_pts) < 3:
            return
        poly = list(self._poly_pts)
        self._poly_mode    = False
        self._poly_pts     = []
        self._display_poly = poly
        self.update()
        self.roi_ready.emit(poly)

    def mouseMoveEvent(self, event) -> None:
        wx = float(event.position().x())
        wy = float(event.position().y())

        self._cursor_pos = (wx, wy)

        # ── Maszk mód egér-mozgás ────────────────────────────────────────────
        if self._smask_mode:
            if self._smask_drag_idx >= 0 and self._img_size is not None:
                # Csúcs húzása
                ic = self._w2i(wx, wy)
                if ic is not None:
                    iw, ih = self._img_size
                    self._smask_poly[self._smask_drag_idx] = (ic[0] / iw, ic[1] / ih)
                    self.smask_poly_changed.emit(list(self._smask_poly))
                    self.update()
            elif self._smask_drawing and self._img_size is not None:
                # Rajzolás közbeni előnézet
                ic = self._w2i(wx, wy)
                if ic is not None:
                    iw, ih = self._img_size
                    self._smask_draw_cursor = (ic[0] / iw, ic[1] / ih)
                    self.update()
            return

        # Sokszög preview, vonallánc preview cursor-vonal → újrarajzolás
        if self._poly_mode or (self._preview_polyline and self._preview_cursor_line):
            self.update()

        # Zoom horgony törlése
        if self._zoom_anchor is not None:
            self._zoom_anchor = None

        # ── ROI drag ─────────────────────────────────────────────────────────
        if self._roi_mode and (event.buttons() & Qt.MouseButton.LeftButton):
            start_ic = self._w2i(self._roi_drag_wx, self._roi_drag_wy)
            curr_ic  = self._w2i(wx, wy)
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
            # Vonallánc csúcs hover frissítés
            new_ph_pidx, new_ph_vidx = self._hit_test_poly_vertex(wx, wy)
            if new_ph_pidx != self._poly_hover_pidx or new_ph_vidx != self._poly_hover_vidx:
                self._poly_hover_pidx = new_ph_pidx
                self._poly_hover_vidx = new_ph_vidx
                self.update()
            if self._poly_mode:
                self.setCursor(Qt.CursorShape.CrossCursor)
            elif handle:
                self.setCursor(self._HANDLE_CURSORS.get(
                    handle, Qt.CursorShape.ArrowCursor))
            elif self._hit_roi_body(wx, wy):
                self.setCursor(Qt.CursorShape.SizeAllCursor)
            elif new_ph_pidx >= 0:
                self.setCursor(Qt.CursorShape.PointingHandCursor)
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
                # Húzás észlelve: klikk-timer törlése (ez drag, nem klikk)
                if self._click_timer.isActive():
                    self._click_timer.stop()
                    self._pending_click_pos = None
            if self._rband_active:
                self._rband_p1 = (wx, wy)
                self.update()
                return

        # ── Vonallánc csúcs húzása (csak 500 ms hosszú nyomás után) ─────────
        if self._poly_drag_pidx >= 0 and (event.buttons() & Qt.MouseButton.LeftButton):
            if self._poly_drag_ready:
                coords = self._w2i(wx, wy)
                if coords is not None:
                    ix, iy = coords
                    self._poly_dragging = True
                    poly = self._polylines_data[self._poly_drag_pidx]
                    if 0 <= self._poly_drag_vidx < len(poly):
                        poly[self._poly_drag_vidx] = (ix, iy)
                    self.update()
                    self.polyline_vertex_moved.emit(
                        self._poly_drag_pidx, self._poly_drag_vidx, ix, iy)
            # return minden esetben: ne induljon gumiszalag vagy pont-mozgatás
            return

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
        # ── Maszk csúcs elengedése ───────────────────────────────────────────
        if event.button() == Qt.MouseButton.LeftButton and self._smask_drag_idx >= 0:
            self._smask_drag_idx = -1
            return

        if event.button() == Qt.MouseButton.LeftButton and self._roi_mode:
            self._roi_mode      = ""
            self._roi_img_start = None
            self.update()
            self.roi_rect_changed.emit(
                QRectF(self._roi_img) if self._roi_img else None)
            return

        if event.button() == Qt.MouseButton.LeftButton:
            if self._rband_active:
                self._finalize_rband()
            elif self._rband_p0 is not None and self._rband_ctrl:
                wx, wy = self._rband_p0
                if self._poly_mode and self._poly_pts:
                    lx, ly = self._poly_pts[-1]
                    if math.hypot(wx - lx, wy - ly) > 5:
                        self._poly_pts.append((wx, wy))
                else:
                    self._poly_mode = True
                    self._poly_pts  = [(wx, wy)]
            self._rband_active         = False
            self._rband_p0             = None
            self._rband_p1             = None
            self._drag_idx             = -1
            self._dragging             = False
            self._poly_longpress_timer.stop()
            self._poly_drag_pidx       = -1
            self._poly_drag_vidx       = -1
            self._poly_dragging        = False
            self._poly_drag_ready      = False
            self.update()

        if event.button() == Qt.MouseButton.RightButton:
            was_click = not self._rpan_moved
            wx_r = float(event.position().x())
            wy_r = float(event.position().y())
            self._rpan_active = False
            self._rpan_moved  = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

            if was_click:
                # Vonallánc csúcs jobb klikk:
                #   sima jobb klikk  → csak ezt a csúcsot törli
                #   Ctrl+jobb klikk  → egész vonalat törli
                ctrl_r = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
                ppidx, pvidx = self._hit_test_poly_vertex(wx_r, wy_r)
                if ppidx >= 0:
                    if ctrl_r:
                        self.polyline_delete_requested.emit(ppidx)
                    else:
                        self.polyline_vertex_delete_requested.emit(ppidx, pvidx)
                    super().mouseReleaseEvent(event)
                    return
                idx = self._hit_test(wx_r, wy_r)
                if idx >= 0:
                    self._selected_set.discard(idx)
                    if self._selected == idx:
                        self._selected = -1
                    self.point_deleted.emit(idx)

        super().mouseReleaseEvent(event)

    def _finalize_rband(self) -> None:
        if self._rband_p0 is None or self._rband_p1 is None:
            return
        x0, y0 = self._rband_p0
        x1, y1 = self._rband_p1
        rx0, rx1 = min(x0, x1), max(x0, x1)
        ry0, ry1 = min(y0, y1), max(y0, y1)
        inside = []
        for i, (ix, iy) in enumerate(self._points):
            wpt = self._i2w(ix, iy)
            if wpt is None:
                continue
            px, py = wpt
            if rx0 <= px <= rx1 and ry0 <= py <= ry1:
                inside.append(i)
        if self._rband_ctrl:
            tl = self._w2i(rx0, ry0)
            br = self._w2i(rx1, ry1)
            if tl is not None and br is not None:
                candidate = QRectF(QPointF(*tl), QPointF(*br)).normalized()
                if candidate.width() >= _ROI_MIN_SIZE and candidate.height() >= _ROI_MIN_SIZE:
                    self._roi_img = candidate
                    self.update()
                    self.roi_rect_changed.emit(QRectF(self._roi_img))
        else:
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
        delta = event.angleDelta().y()
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            n = len(self._points)
            if n == 0:
                event.accept()
                return
            cur = self._selected if 0 <= self._selected < n else 0
            new_idx = (cur - 1) % n if delta > 0 else (cur + 1) % n
            self.point_selected.emit(new_idx)
            event.accept()
        else:
            if self._img_size is None:
                event.accept()
                return
            factor   = self._ZOOM_STEP if delta > 0 else 1.0 / self._ZOOM_STEP
            new_zoom = max(self._ZOOM_MIN, min(self._ZOOM_MAX, self._zoom * factor))
            if new_zoom == self._zoom:
                event.accept()
                return
            if self._zoom_anchor is None:
                mx  = float(event.position().x())
                my  = float(event.position().y())
                hit = self._w2i(mx, my)
                if hit:
                    self._zoom_anchor = hit
                if self._zoom_anchor is None:
                    if 0 <= self._selected < len(self._points):
                        self._zoom_anchor = tuple(self._points[self._selected])  # type: ignore[assignment]
                    else:
                        self._zoom_anchor = (self._pan_cx, self._pan_cy)
            fx, fy       = self._zoom_anchor
            ratio        = self._zoom / new_zoom
            self._pan_cx = fx + (self._pan_cx - fx) * ratio
            self._pan_cy = fy + (self._pan_cy - fy) * ratio
            self._zoom   = new_zoom
            self.update()
            event.accept()

    def keyPressEvent(self, event) -> None:
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
            if self._poly_mode and len(self._poly_pts) >= 3:
                self._close_polygon()
            event.accept()

        elif key == Qt.Key.Key_Escape:
            self.escape_pressed.emit()
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

    def _paint_overlay(self, painter: QPainter) -> None:
        """
        Felülírható hook: a paintEvent VÉGÉN hívódik meg, az összes alap-rajzolás
        befejezése után, de MÉG aktív painter-rel.
        Alosztályok (pl. GCPCanvas) ebben rajzolhatnak rá az alap-rétegre
        anélkül, hogy új QPainter-t kellene létrehozniuk.
        """

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
                f"{self.title}\n\nNincs kép betöltve\n\n"
                "Klikk: pont  |  2× klikk: vonallánc start/vég"
            )
            self._paint_overlay(painter)
            painter.end()
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
            outside = QPainterPath()
            outside.addRect(QRectF(rect))
            inside = QPainterPath()
            inside.addRect(roi_w)
            shadow = outside.subtracted(inside)
            painter.setBrush(QBrush(_C_ROI_OUTSIDE))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPath(shadow)
            painter.setPen(QPen(_C_ROI_BORDER, 2))
            painter.setBrush(QBrush(_C_ROI_FILL))
            painter.drawRect(roi_w)
            painter.setPen(QPen(_C_ROI_BORDER, 1.5))
            painter.setBrush(QBrush(_C_ROI_HANDLE))
            for hx, hy in self._roi_handles_w().values():
                r = _ROI_HANDLE_R
                painter.drawEllipse(int(hx) - r, int(hy) - r, r * 2, r * 2)
            font_roi = QFont()
            font_roi.setPointSize(9)
            painter.setFont(font_roi)
            painter.setPen(QPen(_C_ROI_BORDER, 1))
            lbl = (f"ROI  {int(self._roi_img.width())} × "
                   f"{int(self._roi_img.height())} px")
            painter.drawText(int(roi_w.left()) + 4, int(roi_w.top()) - 5, lbl)

        # ── Vonallánc előnézet overlay ───────────────────────────────────────
        if self._preview_polyline and len(self._preview_polyline) >= 1:
            pts_w = [self._i2w(x, y) for x, y in self._preview_polyline]
            pts_w = [p for p in pts_w if p is not None]
            if pts_w:
                # Összekötő vonalak
                painter.setPen(QPen(_C_POLYLINE, 2))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                for k in range(len(pts_w) - 1):
                    ax, ay = pts_w[k]
                    bx, by = pts_w[k + 1]
                    painter.drawLine(int(ax), int(ay), int(bx), int(by))
                # Szaggatott vonal az egér felé (csak a forrás canvas-on)
                if self._preview_cursor_line:
                    lx, ly = pts_w[-1]
                    cx, cy = self._cursor_pos
                    painter.setPen(QPen(_C_POLYLINE, 1, Qt.PenStyle.DashLine))
                    painter.drawLine(int(lx), int(ly), int(cx), int(cy))
                # Csúcspontok
                painter.setPen(QPen(_C_POLYLINE, 2))
                for k, (px, py) in enumerate(pts_w):
                    r = 7 if k == 0 else 4
                    painter.setBrush(
                        QBrush(_C_POLYLINE) if k == 0 else Qt.BrushStyle.NoBrush)
                    painter.drawEllipse(int(px) - r, int(py) - r, r * 2, r * 2)
                # Tipp szöveg
                if self._preview_cursor_line:
                    painter.setPen(QPen(_C_POLYLINE, 1))
                    font_c = QFont()
                    font_c.setPointSize(9)
                    painter.setFont(font_c)
                    n = len(self._preview_polyline)
                    painter.drawText(
                        6, self.height() - 8,
                        tr(f"Vonallánc ({n} csúcs)  |  "
                           "Klikk: csúcs  |  2× klikk: kész  |  Esc: mégse"))

        # ── Tárolt vonalak (persistent polylines) ───────────────────────────
        _POLY_COLORS = [
            QColor("#22ddcc"), QColor("#ff8a3d"), QColor("#44aaff"),
            QColor("#ff44aa"), QColor("#aaff44"),
        ]
        for pidx, poly in enumerate(self._polylines_data):
            if len(poly) < 2:
                continue
            color = _POLY_COLORS[pidx % len(_POLY_COLORS)]
            pts_w = [self._i2w(float(pt[0]), float(pt[1])) for pt in poly]
            # Összekötő vonalak
            painter.setPen(QPen(color, 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for k in range(len(pts_w) - 1):
                if pts_w[k] is None or pts_w[k + 1] is None:
                    continue
                ax, ay = pts_w[k]
                bx, by = pts_w[k + 1]
                painter.drawLine(int(ax), int(ay), int(bx), int(by))
            # Csúcsok
            for vidx, wpt in enumerate(pts_w):
                if wpt is None:
                    continue
                px, py = wpt
                hover     = (pidx == self._poly_hover_pidx and vidx == self._poly_hover_vidx)
                pressing  = (pidx == self._poly_drag_pidx  and vidx == self._poly_drag_vidx
                             and not self._poly_drag_ready)   # nyomva tartva, 500 ms előtt
                ready     = (pidx == self._poly_drag_pidx  and vidx == self._poly_drag_vidx
                             and self._poly_drag_ready)        # 500 ms lejárt → mozgatható

                if ready:
                    r, alpha, pen_w = 10, 230, 3   # aktívan mozgatható: nagy, vastag
                elif pressing:
                    r, alpha, pen_w = 7, 140, 2    # várakozás: közepes
                elif hover:
                    r, alpha, pen_w = 7, 160, 2    # hover
                else:
                    r, alpha, pen_w = 5, 80,  2    # normál

                fill = QColor(color.red(), color.green(), color.blue(), alpha)
                painter.setPen(QPen(color, pen_w))
                painter.setBrush(QBrush(fill))
                painter.drawEllipse(int(px) - r, int(py) - r, r * 2, r * 2)

                # Kis ikon: 500 ms-t vár → szaggatott külső kör
                if pressing:
                    pen_dash = QPen(color, 1, Qt.PenStyle.DashLine)
                    painter.setPen(pen_dash)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawEllipse(int(px) - r - 4, int(py) - r - 4,
                                        (r + 4) * 2, (r + 4) * 2)
            # Vonallánc sorszám (bal felső sarokba)
            if pts_w[0] is not None:
                fx, fy = pts_w[0]
                font_p = QFont()
                font_p.setPointSize(8)
                painter.setFont(font_p)
                painter.setPen(QPen(color, 1))
                painter.drawText(int(fx) + 9, int(fy) - 5, f"L{pidx + 1}")

        # ── Pontok ───────────────────────────────────────────────────────────
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

        # ── Gumiszalag-keret ─────────────────────────────────────────────────
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
            for k in range(len(pts) - 1):
                ax, ay = pts[k]
                bx, by = pts[k + 1]
                painter.drawLine(int(ax), int(ay), int(bx), int(by))
            if closed and len(pts) >= 3:
                ax, ay = pts[-1]
                bx, by = pts[0]
                painter.drawLine(int(ax), int(ay), int(bx), int(by))
            for k, (px, py) in enumerate(pts):
                if k == 0:
                    r = 6
                    painter.setBrush(QBrush(color))
                else:
                    r = 4
                    painter.setBrush(QBrush(_C_POLY_FILL))
                painter.drawEllipse(int(px) - r, int(py) - r, r * 2, r * 2)

        if self._poly_mode and self._poly_pts:
            _draw_poly(self._poly_pts, _C_POLY_DRAW, closed=False)
            if len(self._poly_pts) >= 1:
                lx, ly = self._poly_pts[-1]
                cx, cy = self._cursor_pos
                painter.setPen(QPen(_C_POLY_DRAW, 1, Qt.PenStyle.DashLine))
                painter.drawLine(int(lx), int(ly), int(cx), int(cy))
            painter.setPen(QPen(QColor("#FFB300"), 1))
            font2 = QFont()
            font2.setPointSize(9)
            painter.setFont(font2)
            n = len(self._poly_pts)
            hint = (f"Ctrl+klikk: csúcs ({n} db)  |  "
                    "Enter / Ctrl+2×klikk: lezár  |  Esc: mégse")
            painter.drawText(6, self.height() - 8, hint)
        elif self._display_poly:
            _draw_poly(self._display_poly, _C_POLY_DONE, closed=True)

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

        # ── Keresési maszk sokszög ───────────────────────────────────────────
        if self._smask_poly and self._img_size is not None:
            iw, ih = self._img_size
            poly_w = []
            for nx, ny in self._smask_poly:
                wpt = self._i2w(nx * iw, ny * ih)
                if wpt:
                    poly_w.append(QPointF(wpt[0], wpt[1]))
            if len(poly_w) >= 2:
                # Kitöltés (csak lezárt sokszög esetén)
                if not self._smask_drawing and len(poly_w) >= 3:
                    path_fill = QPainterPath()
                    path_fill.moveTo(poly_w[0])
                    for p in poly_w[1:]:
                        path_fill.lineTo(p)
                    path_fill.closeSubpath()
                    painter.setBrush(QBrush(QColor(50, 200, 100, 35)))
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.drawPath(path_fill)
                # Kontúr
                pen_style = Qt.PenStyle.DashLine if self._smask_drawing else Qt.PenStyle.SolidLine
                painter.setPen(QPen(QColor(50, 200, 100, 220), 2, pen_style))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                for k in range(len(poly_w) - 1):
                    painter.drawLine(poly_w[k], poly_w[k + 1])
                if not self._smask_drawing and len(poly_w) >= 3:
                    painter.drawLine(poly_w[-1], poly_w[0])
                # Csúcs-körök
                for k, p in enumerate(poly_w):
                    if k == 0 and self._smask_drawing and len(poly_w) >= 3:
                        # Első csúcs sárga kiemelés: ide klikkelve zárul a sokszög
                        painter.setBrush(QBrush(QColor(255, 230, 50, 230)))
                        painter.setPen(QPen(QColor(255, 255, 255), 1.5))
                        painter.drawEllipse(p, 7, 7)
                    else:
                        painter.setBrush(QBrush(QColor(50, 200, 100, 210)))
                        painter.setPen(QPen(QColor(255, 255, 255), 1))
                        painter.drawEllipse(p, 5, 5)
                # Szaggatott vonal az egérig (rajzolás közben)
                if self._smask_drawing and self._smask_draw_cursor and len(poly_w) >= 1:
                    nx, ny = self._smask_draw_cursor
                    cur_w  = self._i2w(nx * iw, ny * ih)
                    if cur_w:
                        painter.setPen(QPen(QColor(50, 200, 100, 130), 1, Qt.PenStyle.DotLine))
                        painter.drawLine(poly_w[-1], QPointF(cur_w[0], cur_w[1]))
                # Felirat ("MASZK" a sokszög fölé)
                if len(poly_w) >= 2:
                    cx = sum(p.x() for p in poly_w) / len(poly_w)
                    cy = min(p.y() for p in poly_w) - 10
                    font_m = QFont()
                    font_m.setPointSize(8)
                    font_m.setBold(True)
                    painter.setFont(font_m)
                    painter.setPen(QPen(QColor(50, 200, 100, 200), 1))
                    lbl = "MASZK (rajzolás…)" if self._smask_drawing else "MASZK"
                    painter.drawText(int(cx) - 30, int(cy), lbl)

        # ── Alosztály-réteg (pl. GCPCanvas overlay) ──────────────────────────
        self._paint_overlay(painter)
        painter.end()


# ────────────────────────────────────────────────────────────────────────────
#  PointEditorWidget
# ────────────────────────────────────────────────────────────────────────────
class PointEditorWidget(QWidget):
    """
    Kétpaneles pontszerkesztő (Kép A + Kép B).

    Interakciós modell
    ------------------
    Egyszeres klikk  → pontpár lerakása (3-pont affin interpoláció a párhoz)
    Dupla klikk      → vonallánc rajzolás start/vég
      közben klikk   → csúcs + azonnali pár a másik képen
    Vonallánc lezáráskor → felosztás 30 px-enként (A-n), interpolált B-pár

    Ha a 3-pont interpoláció hibára fut (pl. kollineáris referencia-pontok),
    hibaablak jelenik meg, az akció törlődik.

    Publikus API
    ------------
    load_images()             – képek és pontok betöltése
    refresh_from_project()    – csak a pontlista frissítése
    undo()                    – visszavonás
    points_changed (signal)   – bármilyen változás után
    """

    points_changed            = pyqtSignal()
    roi_search_requested      = pyqtSignal(list, str, bool, str)
    dual_roi_search_requested = pyqtSignal(object, object, bool, str)
    mask_search_requested     = pyqtSignal(list, list)   # (poly_a_norm, poly_b_norm)
    image_a_drop_requested    = pyqtSignal(str)
    image_b_drop_requested    = pyqtSignal(str)

    _UNDO_LIMIT      = 50
    _POLYLINE_STEP   = 30   # képpont-távolság a vonallánc felosztásánál

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
        self._undo_stack:        List[Tuple[List, List, List]] = []
        self._roi_last_backend:  str  = "SuperPoint + LightGlue"
        self._roi_delete_in_roi: bool = False

        # Vonallánc rajzolás állapota
        self._polyline_active:   bool                          = False
        self._polyline_source:   str                           = ""   # "A" | "B"
        self._polyline_pts_src:  List[Tuple[float, float]]    = []
        self._polyline_pts_dst:  List[Tuple[float, float]]    = []

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
            tr("Pontkeresés és párosítás csak a megjelölt ROI területeken belül"))
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
        self.btn_roi_clear.setStyleSheet(
            "QPushButton{background:#2e2e2e;color:#aaa;padding:0 10px;"
            "border-radius:4px;font-size:12px;}"
            "QPushButton:hover{background:#3a3a3a;}"
            "QPushButton:disabled{background:#2e2e2e;color:#555;}"
        )
        self.btn_roi_clear.clicked.connect(self._clear_rois)

        # ── Maszk gombok ─────────────────────────────────────────────────────
        self.btn_smask = QPushButton(tr("🎭  Maszk"))
        self.btn_smask.setCheckable(True)
        self.btn_smask.setFixedHeight(28)
        self.btn_smask.setToolTip(
            tr("Maszk-sokszög rajzolása\n"
               "Klikk: csúcs hozzáadása  |  Dupla klikk / első csúcsra klikk: lezárás\n"
               "Az egyik képen rajzolt maszk automatikusan megjelenik a másikon is,\n"
               "de mindkét oldalon egymástól függetlenül szerkeszthető"))
        self.btn_smask.setStyleSheet(
            "QPushButton{background:#2a3a2a;color:#aaa;padding:0 10px;"
            "border-radius:4px;font-size:12px;border:1px solid #444;}"
            "QPushButton:checked{background:#1a4a2e;color:#6de89a;border:1px solid #3a8a5a;}"
            "QPushButton:hover{background:#344434;}"
        )
        self.btn_smask.toggled.connect(self._on_smask_mode_toggled)

        self.btn_smask_clear = QPushButton(tr("🗑  Maszk törlése"))
        self.btn_smask_clear.setEnabled(False)
        self.btn_smask_clear.setFixedHeight(28)
        self.btn_smask_clear.setStyleSheet(
            "QPushButton{background:#2e2e2e;color:#aaa;padding:0 10px;"
            "border-radius:4px;font-size:12px;}"
            "QPushButton:hover{background:#3a3a3a;}"
            "QPushButton:disabled{background:#252525;color:#444;}"
        )
        self.btn_smask_clear.clicked.connect(self._clear_smask)

        self.btn_smask_search = QPushButton(tr("🔍  Keresés a maszkban"))
        self.btn_smask_search.setEnabled(False)
        self.btn_smask_search.setFixedHeight(28)
        self.btn_smask_search.setToolTip(
            tr("LightGlue futtatása csak a maszkon belüli területen\n"
               "A talált pontokat hozzáadja a meglévőkhöz"))
        self.btn_smask_search.setStyleSheet(
            "QPushButton{background:#1a3a4a;color:#eee;padding:0 10px;"
            "border-radius:4px;font-size:12px;}"
            "QPushButton:hover{background:#246058;}"
            "QPushButton:disabled{background:#252525;color:#444;}"
        )
        self.btn_smask_search.clicked.connect(self._on_smask_search)

        self.lbl_hint = QLabel(
            tr("Klikk: pont  |  2× klikk: vonallánc start/vég  |  "
               "Húzás: kijelölés  |  Ctrl+húzás: ROI  |  Delete: törlés  |  Görgetés: zoom")
        )
        self.lbl_hint.setStyleSheet("color:#666;font-size:11px;")

        self.lbl_count = QLabel(tr("Pontpárok: 0"))
        self.lbl_count.setStyleSheet("color:#aaa;font-size:12px;")

        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet(
            "color:#22ddcc;font-size:12px;font-weight:bold;")

        bar.addWidget(self.btn_del)
        bar.addSpacing(4)
        bar.addWidget(self.btn_roi_search)
        bar.addWidget(self.btn_roi_clear)
        bar.addSpacing(8)
        bar.addWidget(self.btn_smask)
        bar.addWidget(self.btn_smask_clear)
        bar.addWidget(self.btn_smask_search)
        bar.addSpacing(8)
        bar.addWidget(self.lbl_hint)
        bar.addSpacing(8)
        bar.addWidget(self.lbl_status)
        bar.addStretch()
        bar.addWidget(self.lbl_count)

        # ── Két canvas ───────────────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.canvas_a = PointEditorCanvas(tr("Kép  A"))
        self.canvas_b = PointEditorCanvas(tr("Kép  B"))
        splitter.addWidget(self.canvas_a)
        splitter.addWidget(self.canvas_b)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        root.addLayout(bar)
        root.addWidget(splitter, stretch=1)

        # ── Szignálok ────────────────────────────────────────────────────────
        # Klikk és dupla klikk
        self.canvas_a.single_click_confirmed.connect(
            lambda x, y: self._on_single_click("A", x, y))
        self.canvas_b.single_click_confirmed.connect(
            lambda x, y: self._on_single_click("B", x, y))
        self.canvas_a.double_click_confirmed.connect(
            lambda x, y: self._on_double_click("A", x, y))
        self.canvas_b.double_click_confirmed.connect(
            lambda x, y: self._on_double_click("B", x, y))
        # Esc → vonallánc megszakítása
        self.canvas_a.escape_pressed.connect(self._cancel_polyline)
        self.canvas_b.escape_pressed.connect(self._cancel_polyline)
        # Pont mozgatás / törlés
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
        # ROI keresés
        self.canvas_a.roi_ready.connect(
            lambda poly: self._show_roi_menu(poly, "A"))
        self.canvas_b.roi_ready.connect(
            lambda poly: self._show_roi_menu(poly, "B"))
        # Tartós ROI szinkronizálás
        self.canvas_a.roi_rect_changed.connect(self._on_roi_a_changed)
        self.canvas_b.roi_rect_changed.connect(self._on_roi_b_changed)
        # Kép drag & drop
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
        # Vonallánc csúcs mozgatás / törlés
        self.canvas_a.polyline_vertex_moved.connect(
            lambda pidx, vidx, x, y: self._on_poly_vertex_moved("A", pidx, vidx, x, y))
        self.canvas_b.polyline_vertex_moved.connect(
            lambda pidx, vidx, x, y: self._on_poly_vertex_moved("B", pidx, vidx, x, y))
        self.canvas_a.polyline_delete_requested.connect(self._delete_polyline)
        self.canvas_b.polyline_delete_requested.connect(self._delete_polyline)
        self.canvas_a.polyline_vertex_delete_requested.connect(self._on_poly_vertex_delete)
        self.canvas_b.polyline_vertex_delete_requested.connect(self._on_poly_vertex_delete)

        # Maszk tükrözés: rajzolás befejezésekor a másik canvasra másolja
        self.canvas_a.smask_draw_finished.connect(self._on_smask_a_finished)
        self.canvas_b.smask_draw_finished.connect(self._on_smask_b_finished)
        self.canvas_a.smask_poly_changed.connect(lambda _: self._update_smask_buttons())
        self.canvas_b.smask_poly_changed.connect(lambda _: self._update_smask_buttons())

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
        n = min(len(pts_a), len(pts_b))
        self._selected_set = {i for i in self._selected_set if i < n}
        self.canvas_a.set_selected_set(list(self._selected_set))
        self.canvas_b.set_selected_set(list(self._selected_set))
        self.lbl_count.setText(tr(f"Pontpárok: {n}"))
        self._sync_polylines()

    def _sync_polylines(self) -> None:
        """Tárolt vonalak frissítése mindkét canvas-on."""
        polys_a = [pl.get("pts_a", []) for pl in self.project.polylines]
        polys_b = [pl.get("pts_b", []) for pl in self.project.polylines]
        self.canvas_a.set_polylines(polys_a)
        self.canvas_b.set_polylines(polys_b)

    def _select(self, index: int) -> None:
        self._selected = index
        self.canvas_a.set_selected(index)
        self.canvas_b.set_selected(index)
        has_sel = index >= 0 or bool(self._selected_set)
        self.btn_del.setEnabled(has_sel)

    def _on_selection_a(self, indices: List[int]) -> None:
        self._selected_set = set(indices)
        self.canvas_b.set_selected_set(indices)
        self.btn_del.setEnabled(bool(indices) or self._selected >= 0)

    def _on_selection_b(self, indices: List[int]) -> None:
        self._selected_set = set(indices)
        self.canvas_a.set_selected_set(indices)
        self.btn_del.setEnabled(bool(indices) or self._selected >= 0)

    # ── Undo stack ───────────────────────────────────────────────────────────

    def _push_undo(self) -> None:
        import copy
        state = (
            copy.deepcopy(self.project.anchor_points_a),
            copy.deepcopy(self.project.anchor_points_b),
            copy.deepcopy(self.project.polylines),
        )
        self._undo_stack.append(state)
        if len(self._undo_stack) > self._UNDO_LIMIT:
            self._undo_stack.pop(0)

    def _pop_undo(self) -> bool:
        if not self._undo_stack:
            return False
        state  = self._undo_stack.pop()
        pts_a  = state[0]
        pts_b  = state[1]
        polys  = state[2] if len(state) > 2 else []
        self.project.anchor_points_a[:] = pts_a
        self.project.anchor_points_b[:] = pts_b
        self.project.polylines[:]       = polys
        new_sel = min(self._selected, len(pts_a) - 1)
        self._selected_set.clear()
        self._select(new_sel)
        self._sync()
        self.points_changed.emit()
        return True

    # ── Klikk-esemény kezelés ────────────────────────────────────────────────

    def _on_single_click(self, side: str, x: float, y: float) -> None:
        """Egyszeres klikk: pont lerakása VAGY csúcs hozzáadása aktív vonallánchoz."""
        if self._polyline_active:
            if side != self._polyline_source:
                return  # A másik képre kattintás vonallánc közben: figyelmen kívül
            self._add_polyline_vertex(x, y)
        else:
            self._add_point_pair(side, x, y)

    def _on_double_click(self, side: str, x: float, y: float) -> None:
        """Dupla klikk: vonallánc indítása (ha nem aktív) vagy lezárása (ha aktív)."""
        if self._polyline_active:
            if side != self._polyline_source:
                return
            # Lezárás: utolsó csúcs + commit
            self._add_polyline_vertex(x, y)
            self._commit_polyline()
        else:
            # Ne indítsunk új vonalat, ha meglévő csúcs közelében dupla-klikkelünk
            if not self._near_poly_vertex(side, x, y):
                self._start_polyline(side, x, y)

    def _near_poly_vertex(self, side: str, x: float, y: float,
                          threshold: float = 20.0) -> bool:
        """
        Igaz, ha az (x, y) képkoordináta közelebb van az adott oldal
        valamely vonallánc-csúcsához, mint `threshold` pixel.
        """
        key = "pts_a" if side == "A" else "pts_b"
        for poly in self.project.polylines:
            for pt in poly.get(key, []):
                if math.hypot(x - pt[0], y - pt[1]) <= threshold:
                    return True
        return False

    # ── Pontpár lerakása (egyszeres klikk) ──────────────────────────────────

    def _add_point_pair(self, side: str, x: float, y: float) -> None:
        """
        Egy pontpár hozzáadása: az egyik kép (x,y) koordinátájára klikkelve
        a másik kép megfelelő helyzetét 3-pont affin interpolációval számolja.
        """
        pts_src = (self.project.anchor_points_a if side == "A"
                   else self.project.anchor_points_b)
        pts_dst = (self.project.anchor_points_b if side == "A"
                   else self.project.anchor_points_a)
        try:
            partner = self._interpolate_partner(x, y, pts_src, pts_dst)
        except ValueError as e:
            self._show_error(str(e))
            return

        self._push_undo()
        if side == "A":
            self.project.anchor_points_a.append([x, y])
            self.project.anchor_points_b.append(list(partner))
        else:
            self.project.anchor_points_b.append([x, y])
            self.project.anchor_points_a.append(list(partner))

        new_idx = len(self.project.anchor_points_a) - 1
        self._selected_set.clear()
        self._select(new_idx)
        self._sync()
        self.points_changed.emit()

    # ── Vonallánc rajzolás ───────────────────────────────────────────────────

    def _start_polyline(self, side: str, x: float, y: float) -> None:
        """Vonallánc rajzolás indítása – első csúcs lerakása."""
        pts_src = (self.project.anchor_points_a if side == "A"
                   else self.project.anchor_points_b)
        pts_dst = (self.project.anchor_points_b if side == "A"
                   else self.project.anchor_points_a)
        try:
            partner = self._interpolate_partner(x, y, pts_src, pts_dst)
        except ValueError as e:
            self._show_error(str(e))
            return

        self._polyline_active  = True
        self._polyline_source  = side
        self._polyline_pts_src = [(x, y)]
        self._polyline_pts_dst = [partner]
        self._update_polyline_preview()
        self.lbl_status.setText(
            tr(f"Vonallánc  ({side})  |  Klikk: csúcs  |  2× klikk: kész  |  Esc: mégse"))

    def _add_polyline_vertex(self, x: float, y: float) -> None:
        """Csúcs hozzáadása az aktív vonallánchoz; párja azonnal megjelenik."""
        side    = self._polyline_source
        pts_src = (self.project.anchor_points_a if side == "A"
                   else self.project.anchor_points_b)
        pts_dst = (self.project.anchor_points_b if side == "A"
                   else self.project.anchor_points_a)
        try:
            partner = self._interpolate_partner(x, y, pts_src, pts_dst)
        except ValueError as e:
            self._show_error(str(e))
            return

        self._polyline_pts_src.append((x, y))
        self._polyline_pts_dst.append(partner)
        self._update_polyline_preview()

    def _commit_polyline(self) -> None:
        """
        Vonallánc lezárva: eltárolja a csúcsokat polyline objektumként.
        A felosztás (30 px) csak rendereléskor történik – nem most.
        """
        if len(self._polyline_pts_src) < 2:
            self._cancel_polyline()
            return

        side = self._polyline_source
        self._push_undo()

        if side == "A":
            entry = {
                "pts_a": [[x, y] for x, y in self._polyline_pts_src],
                "pts_b": [[x, y] for x, y in self._polyline_pts_dst],
            }
        else:
            entry = {
                "pts_a": [[x, y] for x, y in self._polyline_pts_dst],
                "pts_b": [[x, y] for x, y in self._polyline_pts_src],
            }

        self.project.polylines.append(entry)
        self._cancel_polyline()   # preview törlése
        self._sync_polylines()
        self.points_changed.emit()

    def _cancel_polyline(self) -> None:
        """Vonallánc megszakítása – minden előnézet törlése."""
        self._polyline_active  = False
        self._polyline_source  = ""
        self._polyline_pts_src = []
        self._polyline_pts_dst = []
        self.canvas_a.set_preview_polyline(None)
        self.canvas_b.set_preview_polyline(None)
        self.lbl_status.setText("")

    def _update_polyline_preview(self) -> None:
        """Előnézet frissítése: forrás canvas-on szaggatott vonal, cél canvas-on pontok."""
        if self._polyline_source == "A":
            self.canvas_a.set_preview_polyline(
                self._polyline_pts_src, show_cursor_line=True)
            self.canvas_b.set_preview_polyline(
                self._polyline_pts_dst, show_cursor_line=False)
        else:
            self.canvas_b.set_preview_polyline(
                self._polyline_pts_src, show_cursor_line=True)
            self.canvas_a.set_preview_polyline(
                self._polyline_pts_dst, show_cursor_line=False)

    # ── Affin interpoláció ────────────────────────────────────────────────────

    _INTERP_CANDIDATES = 8   # hány legközelebbi pontból keresünk jó háromszöget
    _COND_LIMIT        = 1e10  # ennél nagyobb kondíciószámú háromszög "rossz"

    def _interpolate_partner(self,
                              x: float, y: float,
                              src_pts: list, dst_pts: list
                              ) -> Tuple[float, float]:
        """
        A (x, y) forrás-koordináta párját kiszámítja a meglévő pontpárok alapján.

        Algoritmus:
          1. A K (≤ 8) legközelebbi forráspontból kiválasztja a legjobb
             kondíciószámú háromszöget (C(K,3) kombináció átvizsgálásával).
          2. Ha minden háromszög közel kollineáris (cond > 1e10), tartalékként
             súlyozott legkisebb-négyzetek affin illesztést végez az összes
             ponton (közelebbi pontok nagyobb súlyt kapnak).

        Kivétel: ValueError – csak akkor, ha n < 3.
        """
        from itertools import combinations

        n = len(src_pts)
        if n < 3:
            raise ValueError(
                tr(f"Nincs elég referenciapont az interpolációhoz "
                   f"({n} db, minimum 3 kell).\n"
                   "A GCP igazítás után legalább 3 pontpárnak kell lennie."))

        # ── 1. lépés: legjobb háromszög keresése a K legközelebbi pontból ────
        K = min(self._INTERP_CANDIDATES, n)
        dists = sorted(
            (math.hypot(x - float(p[0]), y - float(p[1])), i)
            for i, p in enumerate(src_pts)
        )
        cands = [dists[k][1] for k in range(K)]

        best_cond    = float("inf")
        best_triplet = None

        for i1, i2, i3 in combinations(cands, 3):
            p1, p2, p3 = src_pts[i1], src_pts[i2], src_pts[i3]
            A = np.array([
                [float(p1[0]), float(p1[1]), 1.0],
                [float(p2[0]), float(p2[1]), 1.0],
                [float(p3[0]), float(p3[1]), 1.0],
            ], dtype=np.float64)
            cond = np.linalg.cond(A)
            if cond < best_cond:
                best_cond    = cond
                best_triplet = (i1, i2, i3)

        if best_cond <= self._COND_LIMIT and best_triplet is not None:
            i1, i2, i3 = best_triplet
            p1, p2, p3 = src_pts[i1], src_pts[i2], src_pts[i3]
            q1, q2, q3 = dst_pts[i1], dst_pts[i2], dst_pts[i3]
            A = np.array([
                [float(p1[0]), float(p1[1]), 1.0],
                [float(p2[0]), float(p2[1]), 1.0],
                [float(p3[0]), float(p3[1]), 1.0],
            ], dtype=np.float64)
            cu = np.linalg.solve(
                A, [float(q1[0]), float(q2[0]), float(q3[0])])
            cv = np.linalg.solve(
                A, [float(q1[1]), float(q2[1]), float(q3[1])])
            return (float(cu[0]*x + cu[1]*y + cu[2]),
                    float(cv[0]*x + cv[1]*y + cv[2]))

        # ── 2. lépés tartalék: súlyozott legkisebb négyzetek ─────────────────
        return self._interpolate_weighted(x, y, src_pts, dst_pts)

    @staticmethod
    def _interpolate_weighted(x: float, y: float,
                               src_pts: list, dst_pts: list
                               ) -> Tuple[float, float]:
        """
        Súlyozott legkisebb-négyzetek affin illesztés.
        Minden meglévő pontpárt felhasznál; a közelebb lévők nagyobb súlyt kapnak.
        Kollineáris elrendezés esetén is robusztusan működik.
        """
        src = np.array([[float(p[0]), float(p[1])] for p in src_pts], dtype=np.float64)
        dst = np.array([[float(p[0]), float(p[1])] for p in dst_pts], dtype=np.float64)

        # 1 / (távolság + ε) alapú súlyok  →  közelebbi pontok dominálnak
        dists   = np.sqrt((src[:, 0] - x) ** 2 + (src[:, 1] - y) ** 2)
        weights = 1.0 / (dists + 1e-6)

        A  = np.column_stack([src, np.ones(len(src))])   # [n × 3]
        W  = np.diag(weights)
        WA = W @ A

        cu, *_ = np.linalg.lstsq(WA, W @ dst[:, 0], rcond=None)
        cv, *_ = np.linalg.lstsq(WA, W @ dst[:, 1], rcond=None)

        return (float(cu[0]*x + cu[1]*y + cu[2]),
                float(cv[0]*x + cv[1]*y + cv[2]))

    # ── Vonallánc felosztás (30 px-enként a forrás oldalon) ──────────────────

    @staticmethod
    def _subdivide_by_distance(pts: list, step: float) -> list:
        """
        Pontokat mintavételez egy töröttvonal mentén, minden `step` képpont után egyet.
        Az első pontot mindig tartalmazza; az utolsót is, ha nincs pont közelében.
        """
        if len(pts) < 2:
            return [tuple(pts[0])] if pts else []

        result = [tuple(pts[0])]
        carry  = 0.0   # az előző szakaszból áthozott töredék-távolság

        for i in range(1, len(pts)):
            dx = pts[i][0] - pts[i - 1][0]
            dy = pts[i][1] - pts[i - 1][1]
            seg_len = math.hypot(dx, dy)
            if seg_len < 1e-9:
                continue

            # Első mintavételi pont a szakaszon belül (step - carry távolságra az elejétől)
            dist_in_seg = step - carry
            while dist_in_seg <= seg_len + 1e-9:
                t = min(1.0, dist_in_seg / seg_len)
                result.append((
                    pts[i - 1][0] + t * dx,
                    pts[i - 1][1] + t * dy,
                ))
                dist_in_seg += step

            # Maradék távolság (a következő szakaszra viszi át)
            carry = seg_len - (dist_in_seg - step)

        # Utolsó pont hozzáadása, ha nincs közel
        last = result[-1]
        end  = pts[-1]
        if math.hypot(last[0] - end[0], last[1] - end[1]) > 1.0:
            result.append(tuple(end))

        return result

    # ── Hibajelzés ──────────────────────────────────────────────────────────

    def _show_error(self, message: str) -> None:
        """Hibaablak megjelenítése (OK gombbal)."""
        dlg = QMessageBox(self)
        dlg.setWindowTitle(tr("ArchMorph – Hiba"))
        dlg.setText(message)
        dlg.setIcon(QMessageBox.Icon.Warning)
        dlg.setStyleSheet(
            "QMessageBox{background:#1e2430;color:#eee;}"
            "QLabel{color:#eee;font-size:13px;}"
            "QPushButton{background:#2a3040;color:#eee;padding:4px 20px;"
            "border-radius:4px;border:1px solid #444;}"
            "QPushButton:hover{background:#3a4050;}"
        )
        dlg.exec()

    # ── Tartós téglalap-ROI kezelés ──────────────────────────────────────────

    def _on_roi_a_changed(self, roi_img) -> None:
        self._update_roi_buttons()
        if roi_img is None:
            return
        img_a = self.project.image_a
        img_b = self.project.image_b
        if img_a is None or img_b is None:
            return
        ha, wa = img_a.shape[:2]
        hb, wb = img_b.shape[:2]
        ax = roi_img.left()  / wa;  ay = roi_img.top()    / ha
        aw = roi_img.width() / wa;  ah = roi_img.height() / ha
        b_rect = QRectF(ax * wb, ay * hb, aw * wb, ah * hb)
        self.canvas_b.roi_rect_changed.disconnect(self._on_roi_b_changed)
        self.canvas_b.set_roi_from_image(b_rect)
        self.canvas_b.roi_rect_changed.connect(self._on_roi_b_changed)
        self._update_roi_buttons()

    def _on_roi_b_changed(self, roi_img) -> None:
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

    # ── Maszk metódusok ──────────────────────────────────────────────────────

    def _on_smask_mode_toggled(self, checked: bool) -> None:
        self.canvas_a.set_smask_mode(checked)
        self.canvas_b.set_smask_mode(checked)

    def _on_smask_a_finished(self, pts: list) -> None:
        """A kép maszkjának befejezésekor automatikusan tükrözi B-re."""
        self.canvas_b.set_smask_poly(pts)
        self._update_smask_buttons()

    def _on_smask_b_finished(self, pts: list) -> None:
        """B kép maszkjának befejezésekor automatikusan tükrözi A-ra."""
        self.canvas_a.set_smask_poly(pts)
        self._update_smask_buttons()

    def _clear_smask(self) -> None:
        self.canvas_a.clear_smask()
        self.canvas_b.clear_smask()
        self._update_smask_buttons()

    def _update_smask_buttons(self) -> None:
        has_a = len(self.canvas_a._smask_poly) >= 3
        has_b = len(self.canvas_b._smask_poly) >= 3
        self.btn_smask_clear.setEnabled(has_a or has_b)
        self.btn_smask_search.setEnabled(has_a and has_b)

    def _on_smask_search(self) -> None:
        poly_a = list(self.canvas_a._smask_poly)
        poly_b = list(self.canvas_b._smask_poly)
        if len(poly_a) >= 3 and len(poly_b) >= 3:
            self.mask_search_requested.emit(poly_a, poly_b)

    def get_smask_polys(self):
        """Visszaadja a két maszk sokszöget norm. koordinátákban, vagy (None, None)."""
        pa = list(self.canvas_a._smask_poly) if len(self.canvas_a._smask_poly) >= 3 else None
        pb = list(self.canvas_b._smask_poly) if len(self.canvas_b._smask_poly) >= 3 else None
        return pa, pb

    def _on_dual_roi_search(self) -> None:
        roi_a = self.canvas_a.get_roi_image()
        roi_b = self.canvas_b.get_roi_image()
        if roi_a is None or roi_b is None:
            return
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
        act_go = QAction(tr("🔍  Keresés indítása"), menu)
        menu.addAction(act_go)
        chosen = menu.exec(self.btn_roi_search.mapToGlobal(
            self.btn_roi_search.rect().bottomLeft()))
        if chosen is None or chosen is not act_go:
            return
        for act in ag.actions():
            if act.isChecked():
                self._roi_last_backend = act.text()
                break
        self._roi_delete_in_roi = act_del.isChecked()
        self.dual_roi_search_requested.emit(
            roi_a, roi_b, self._roi_delete_in_roi, self._roi_last_backend)

    # ── ROI helyi menü ───────────────────────────────────────────────────────

    def _show_roi_menu(self, widget_poly: list, side: str) -> None:
        canvas = self.canvas_a if side == "A" else self.canvas_b
        img_poly = []
        for wx, wy in widget_poly:
            ic = canvas._w2i(wx, wy)
            if ic is not None:
                img_poly.append([float(ic[0]), float(ic[1])])
            else:
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
        canvas._display_poly = list(widget_poly)
        canvas.update()
        menu = QMenu(canvas)
        menu.setStyleSheet(self._MENU_STYLE)
        n_verts    = len(img_poly)
        shape_name = "téglalap-ROI" if n_verts == 4 else f"{n_verts}-szög ROI"
        title_act  = QAction(tr(f"  {shape_name}  –  keresési beállítások"), menu)
        title_act.setEnabled(False)
        menu.addAction(title_act)
        menu.addSeparator()
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
        del_act = QAction(tr("Meglévő pontok törlése a területen belül"), menu)
        del_act.setCheckable(True)
        del_act.setChecked(self._roi_delete_in_roi)
        menu.addAction(del_act)
        menu.addSeparator()
        search_act = QAction(tr("▶  ROI keresés indítása"), menu)
        menu.addAction(search_act)
        cancel_act = QAction(tr("✕  Mégse"), menu)
        menu.addAction(cancel_act)
        last_wx, last_wy = widget_poly[-1]
        global_pos = canvas.mapToGlobal(QPoint(int(last_wx), int(last_wy)))
        chosen     = menu.exec(global_pos)
        if chosen in bk_acts:
            self._roi_last_backend = bk_acts[chosen]
            self._show_roi_menu(widget_poly, side)
            return
        if chosen == del_act:
            self._roi_delete_in_roi = del_act.isChecked()
            self._show_roi_menu(widget_poly, side)
            return
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

    # ── Vonallánc csúcs mozgatás / törlés ────────────────────────────────────

    def _on_poly_vertex_moved(self, side: str,
                               pidx: int, vidx: int,
                               x: float, y: float) -> None:
        """Vonallánc csúcs mozgatva – projekt frissítése (canvas már friss)."""
        if 0 <= pidx < len(self.project.polylines):
            key  = "pts_a" if side == "A" else "pts_b"
            poly = self.project.polylines[pidx]
            pts  = poly.get(key, [])
            if 0 <= vidx < len(pts):
                pts[vidx] = [x, y]
            self.points_changed.emit()

    def _delete_polyline(self, pidx: int) -> None:
        """Vonallánc törlése (Ctrl+jobb klikk)."""
        if 0 <= pidx < len(self.project.polylines):
            self._push_undo()
            self.project.polylines.pop(pidx)
            self._sync_polylines()
            self.points_changed.emit()

    def _on_poly_vertex_delete(self, pidx: int, vidx: int) -> None:
        """
        Egy csúcs törlése a vonalláncból (sima jobb klikk csúcson).
        Ha a törlés után kevesebb mint 2 csúcs marad, az egész vonallánc törlődik.
        """
        if 0 <= pidx < len(self.project.polylines):
            poly  = self.project.polylines[pidx]
            pts_a = poly.get("pts_a", [])
            pts_b = poly.get("pts_b", [])
            n     = min(len(pts_a), len(pts_b))
            if 0 <= vidx < n:
                self._push_undo()
                pts_a.pop(vidx)
                pts_b.pop(vidx)
                # Ha 1 csúcs vagy kevesebb maradt → egész vonallánc törlése
                if min(len(pts_a), len(pts_b)) < 2:
                    self.project.polylines.pop(pidx)
                self._sync_polylines()
                self.points_changed.emit()

    # ── Renderelési segéd: vonalak → pontpárok ───────────────────────────────

    @staticmethod
    def polylines_to_point_pairs(polylines: list,
                                  step: float = 30.0,
                                  ) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Rendereléskor hívandó: minden tárolt vonallánc felosztása pontpárokká.

        Algoritmus:
          1. A-oldali vonalat `step` px-enként osztja → N pontot kap
          2. B-oldali vonalat pontosan N pontra mintavételezi (egyenletesen)
          3. Visszaadja az összes pontpárt

        Visszatér: (pts_a_all, pts_b_all)
        """
        pts_a_all: List[List[float]] = []
        pts_b_all: List[List[float]] = []

        for poly in polylines:
            src = poly.get("pts_a", [])
            dst = poly.get("pts_b", [])
            if len(src) < 2 or len(dst) < 2:
                continue
            sub_a = PointEditorWidget._subdivide_by_distance(src, step)
            n = len(sub_a)
            if n < 2:
                continue
            sub_b = PointEditorWidget._resample_to_n(dst, n)
            pts_a_all.extend([[pt[0], pt[1]] for pt in sub_a])
            pts_b_all.extend([[pt[0], pt[1]] for pt in sub_b])

        return pts_a_all, pts_b_all

    @staticmethod
    def _resample_to_n(pts: list, n: int) -> list:
        """
        Töröttvonalat pontosan `n` pontra mintavételez (egyenletesen hossz szerint).
        Az első és az utolsó pont mindig szerepel.
        """
        if n <= 1:
            return [tuple(pts[0])]

        # Kumulatív hossz
        lengths = [0.0]
        for i in range(1, len(pts)):
            d = math.hypot(
                float(pts[i][0]) - float(pts[i - 1][0]),
                float(pts[i][1]) - float(pts[i - 1][1]),
            )
            lengths.append(lengths[-1] + d)

        total = lengths[-1]
        if total < 1e-9:
            return [tuple(pts[0])] * n

        result = []
        for k in range(n):
            target = total * k / (n - 1)
            # Szakasz keresése
            for i in range(1, len(lengths)):
                if lengths[i] >= target - 1e-9 or i == len(lengths) - 1:
                    seg_len = lengths[i] - lengths[i - 1]
                    if seg_len < 1e-9:
                        result.append((float(pts[i][0]), float(pts[i][1])))
                    else:
                        t = min(1.0, max(0.0, (target - lengths[i - 1]) / seg_len))
                        x = float(pts[i - 1][0]) + t * (float(pts[i][0]) - float(pts[i - 1][0]))
                        y = float(pts[i - 1][1]) + t * (float(pts[i][1]) - float(pts[i - 1][1]))
                        result.append((x, y))
                    break

        return result

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
        if self._selected_set:
            self._push_undo()
            pa = self.project.anchor_points_a
            pb = self.project.anchor_points_b
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
