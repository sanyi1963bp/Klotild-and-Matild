"""
gcp_dialog.py  –  GCP (Ground Control Points) jelölő párbeszédablak
====================================================================
A GCPCanvas a PointEditorCanvas-ból örököl, ezért azonos rajzolási
eszközök állnak rendelkezésre, mint a kézi pontszerkesztőben:

  Pont mód   : kattints A képen → kattints B képen (alternáló)
  Vonallánc  : bal gomb + húzás = szakasz, 2× klikk = kész
               → N pontpár keletkezik (felosztás spin-boxból)
  Ív mód     : klikk: kezdőpont → végpont → köztes pontok → 2× klikk kész
               → N pontpár keletkezik (alapértelmezett 12)

Minden módban:
  Jobb gomb + húzás        → kép pásztázás
  Görgetés                 → zoom
  Bal gomb + húzás ponton  → pont finomhangolás
  Jobb klikk ponton        → pár törlése (pont mód) / utolsó csúcs
                             visszavonása (görbe mód)
  Backspace / Esc          → görbe módban visszavonás / megszakítás

Használat
---------
    dlg = GCPDialog(img_a_bgr, img_b_bgr, parent)
    if dlg.exec() == QDialog.DialogCode.Accepted:
        pairs = dlg.get_pairs()   # [(pt_a, pt_b), ...]
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QSpinBox, QVBoxLayout, QWidget,
)

from point_editor import PointEditorCanvas, PointEditorWidget

try:
    from TRANSLATIONS import tr
except ImportError:
    def tr(text: str) -> str: return text


# ────────────────────────────────────────────────────────────────────────────
#  GCPCanvas  –  PointEditorCanvas alapú GCP-canvas
# ────────────────────────────────────────────────────────────────────────────

class GCPCanvas(PointEditorCanvas):
    """
    GCP-specifikus canvas.

    PointEditorCanvas-tól örökölt főbb funkciók:
      - Pan / zoom (jobb gomb + húzás, görgetés)
      - Vonallánc rajzolás (press+drag+release = szakasz, 2× klikk = kész)
      - Ív rajzolás (klikk: start → end → köztes → 2× klikk kész)
      - Pont húzás (bal gomb meglévő ponton)
      - curve_done, point_moved, point_deleted szignálok

    GCP-specifikus kiegészítések:
      - point_clicked(ix, iy)  : pont módban egyszeres bal klikk jele
      - set_active(bool)       : fogad-e ÚJ bemenetet
      - set_pending(pt | None) : félkész A-pont megjelenítése (B-t vár)
      - paintEvent             : aktív keret + pending pont + oldal-cimke
    """

    point_clicked = pyqtSignal(float, float)
    # Stub: a régi PointEditorCanvas-ban létező görbe-jel emulációja.
    # Az új tervben a GCP dialog csak pont-módban működik; a vonallánc/ív
    # gombok megtartva a UI-ban, de nem csinálnak semmit.
    curve_done    = pyqtSignal(list)

    def __init__(self, side: str) -> None:
        super().__init__(side)          # title = "A" / "B"
        self._active     = False
        self._draw_mode  = "point"      # "point" | "polyline" | "arc"  (jelenleg csak "point" aktív)
        self._pending: Optional[Tuple[float, float]] = None

    # ── Publikus API ─────────────────────────────────────────────────────────

    def set_active(self, active: bool) -> None:
        """Fogad-e ÚJ bemenetet (klikk / görberajzolás)."""
        self._active = active
        self.update()

    def set_pending(self, pt: Optional[Tuple[float, float]]) -> None:
        """Félkész A-pont megjelenítés (szaggatott kör); None = nincs."""
        self._pending = pt
        self.update()

    def set_draw_mode(self, mode: str) -> None:
        """Rajzolási mód beállítása (stub – jelenleg csak 'point' aktív)."""
        self._draw_mode = mode

    def cancel_curve(self) -> None:
        """Félkész görbe törlése (stub – görbe-funkció nem aktív)."""
        pass

    # ── Esemény-felülírások ──────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
            wx = float(event.position().x())
            wy = float(event.position().y())

            if not ctrl:
                if self._draw_mode == "point":
                    # Aktív + üres hely → GCP pont egyszeres klikkre
                    if self._active and self._hit_test(wx, wy) < 0:
                        coords = self._w2i(wx, wy)
                        if coords is not None:
                            self.point_clicked.emit(*coords)
                        return
                    # Meglévő pont → szülő kezeli a húzást (aktív vagy nem)
                elif self._draw_mode in ("polyline", "arc"):
                    # Görbe módban inaktív canvas nem rajzol
                    if not self._active:
                        return

        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # type: ignore[override]
        # Görbe mód + inaktív canvas → kettős klikk sem csinál semmit
        if (event.button() == Qt.MouseButton.LeftButton
                and self._draw_mode in ("polyline", "arc")
                and not self._active):
            return
        super().mouseDoubleClickEvent(event)

    # ── GCP-specifikus rajzolás – hook a szülő painter-ébe ───────────────────

    def _paint_overlay(self, p: "QPainter") -> None:  # type: ignore[override]
        """
        A PointEditorCanvas.paintEvent() az összes alap-rajzolás után
        egy aktív painter-rel hívja meg ezt a metódust.
        Így egyetlen QPainter van aktív egyszerre – elkerüljük a
        "device already being painted" Qt-összeomlást.
        """
        p.setRenderHint(p.RenderHint.Antialiasing)

        # ── GCP pontok kontrasztos újrarajzolása ─────────────────────────────
        # A szülő már rajzolt narancsos köröket, de világos háttér esetén
        # nem látszanak. Felülrajzoljuk: fekete kontúr + élénk szín belül.
        for i, pt in enumerate(self._points):
            wpt = self._i2w(*pt)
            if wpt is None:
                continue
            px, py = int(wpt[0]), int(wpt[1])
            r = 8

            # Fekete körvonal (kontraszt réteg)
            p.setPen(QPen(QColor(0, 0, 0, 220), 3))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(px - r, py - r, r * 2, r * 2)

            # Élénk belső töltés (narancssárga)
            p.setPen(QPen(QColor("#ff8a3d"), 2))
            p.setBrush(QBrush(QColor(255, 138, 61, 160)))
            p.drawEllipse(px - r + 1, py - r + 1, (r - 1) * 2, (r - 1) * 2)

            # Sorszám fehér, fekete árnyékkal
            font = QFont("Arial", 8, QFont.Weight.Bold)
            p.setFont(font)
            tx, ty = px + r + 2, py + 4
            # árnyék
            p.setPen(QPen(QColor(0, 0, 0, 200), 1))
            p.drawText(tx + 1, ty + 1, str(i + 1))
            # szöveg
            p.setPen(QPen(QColor("#ffffff"), 1))
            p.drawText(tx, ty, str(i + 1))

        # ── Aktív keret (ciánkék) ────────────────────────────────────────────
        if self._active:
            p.setPen(QPen(QColor("#22ddcc"), 3))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(2, 2, self.width() - 4, self.height() - 4)

        # ── Pending pont (párjára vár) ───────────────────────────────────────
        if self._pending is not None:
            wpt = self._i2w(*self._pending)
            if wpt is not None:
                px, py = int(wpt[0]), int(wpt[1])
                # Fekete körvonal
                p.setPen(QPen(QColor(0, 0, 0, 180), 3))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(px - 12, py - 12, 24, 24)
                # Szaggatott sárga
                p.setPen(QPen(QColor("#ffe060"), 2, Qt.PenStyle.DashLine))
                p.setBrush(QBrush(QColor(255, 224, 96, 50)))
                p.drawEllipse(px - 11, py - 11, 22, 22)

        # ── Oldal-cimke (A / B) – bal felső sarok ───────────────────────────
        p.fillRect(8, 8, 34, 26, QColor(0, 0, 0, 200))
        p.setPen(QPen(QColor("#ffffff"), 1))
        font = QFont("Arial", 13, QFont.Weight.Bold)
        p.setFont(font)
        p.drawText(12, 28, self.title)


# ────────────────────────────────────────────────────────────────────────────
#  GCPDialog  –  a teljes párbeszédablak
# ────────────────────────────────────────────────────────────────────────────

class GCPDialog(QDialog):
    """
    Ground Control Points jelölő dialóg.

    Workflow pont módban:
      1. Kattints az A képen (bal) egy jól azonosítható pontra.
      2. Kattints a megfelelő pontra a B képen (jobb).
      3. Ismételd 4–15-ször.
      4. „Igazítás számítása" → accept().

    Workflow vonallánc módban:
      1. Húzz vonalláncot az A képen (szakaszok: bal gomb + húzás + elenged).
         Kettős klikk = kész.
      2. Húzz megfelelő vonalláncot a B képen.
      3. A rendszer N egyenközű pontpárt generál a két vonallánc mentén.

    Workflow ív módban:
      1. Kattints az A képen: kezdőpont → végpont → köztes pontok → 2× klikk kész.
      2. Kattints a B képen ugyanúgy.
      3. A rendszer N egyenközű pontpárt generál a két körív mentén.

    get_pairs() → [(pt_a, pt_b), ...]  ahol pt = (x_float, y_float) képkoordban.
    """

    def __init__(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("GCP-k megjelölése – geometriai igazítás")
        self.setMinimumSize(960, 600)
        self.resize(1200, 740)
        self.setModal(True)

        self._img_a = img_a
        self._img_b = img_b

        # Pár-lista: [(pt_a, pt_b), ...]
        self._pairs: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        # Félkész A-pont (pont módban, B-párra vár)
        self._pending_a: Optional[Tuple[float, float]] = None
        # Félkész A-görbe (görbe módban, B-görbére vár)
        self._pending_curve_a: Optional[list] = None

        self._next_side = "A"
        self._draw_mode = "point"       # "point" | "polyline" | "arc"

        self._build_ui()
        self.canvas_a.set_image(img_a)
        self.canvas_b.set_image(img_b)
        self._set_mode("point")         # kezdeti mód + gomb-stílusok

    # ── UI építés ────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(5)

        # ── Mód-választó sáv ─────────────────────────────────────────────────
        mode_row = QHBoxLayout()
        mode_row.setSpacing(4)

        mode_lbl = QLabel(tr("Eszköz:"))
        mode_lbl.setStyleSheet("color:#88aacc; font-size:12px;")
        mode_row.addWidget(mode_lbl)

        self._btn_mode_pt  = QPushButton(tr("● Pont"))
        self._btn_mode_pl  = QPushButton(tr("⟋ Vonallánc"))
        self._btn_mode_arc = QPushButton(tr("⌒ Ív"))
        self._btn_mode_pt.setToolTip(
            tr("Pont mód: kattints A képen, majd a párjára B képen"))
        self._btn_mode_pl.setToolTip(
            tr("Vonallánc: bal gomb+húzás = szakasz, 2× klikk = kész\n"
               "Előbb A képen, majd B képen → N pontpár keletkezik"))
        self._btn_mode_arc.setToolTip(
            tr("Ív: klikk start → klikk end → köztes pontok → 2× klikk kész\n"
               "Előbb A képen, majd B képen → N pontpár keletkezik"))
        for btn in (self._btn_mode_pt, self._btn_mode_pl, self._btn_mode_arc):
            mode_row.addWidget(btn)

        self._btn_mode_pt .clicked.connect(lambda: self._set_mode("point"))
        self._btn_mode_pl .clicked.connect(lambda: self._set_mode("polyline"))
        self._btn_mode_arc.clicked.connect(lambda: self._set_mode("arc"))

        # Felosztás spinbox (vonallánc / ív módban releváns)
        sep = QLabel("  |")
        sep.setStyleSheet("color:#445;")
        mode_row.addWidget(sep)

        n_lbl = QLabel(tr("N pont:"))
        n_lbl.setStyleSheet("color:#88aacc; font-size:12px;")
        mode_row.addWidget(n_lbl)

        self._spin_n = QSpinBox()
        self._spin_n.setRange(3, 100)
        self._spin_n.setValue(12)
        self._spin_n.setFixedWidth(60)
        self._spin_n.setToolTip(
            tr("Vonallánc/ív mentén generált pontpárok száma"))
        self._spin_n.setStyleSheet(
            "QSpinBox{background:#2a2a3a; color:#ccc; border:1px solid #445;}")
        mode_row.addWidget(self._spin_n)

        mode_row.addStretch()
        root.addLayout(mode_row)

        # ── Instrukció sáv ───────────────────────────────────────────────────
        self._lbl_instr = QLabel()
        self._lbl_instr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_instr.setStyleSheet(
            "color:#aaddff; font-size:13px; padding:5px 10px;"
            "background:#1a2a3a; border-radius:4px;")
        root.addWidget(self._lbl_instr)

        # ── Két canvas egymás mellé ──────────────────────────────────────────
        canvases_row = QHBoxLayout()
        canvases_row.setSpacing(4)
        self.canvas_a = GCPCanvas("A")
        self.canvas_b = GCPCanvas("B")
        canvases_row.addWidget(self.canvas_a)
        canvases_row.addWidget(self.canvas_b)
        root.addLayout(canvases_row, stretch=1)

        # ── Vezérlő sor ──────────────────────────────────────────────────────
        ctrl = QHBoxLayout()
        ctrl.setSpacing(8)

        self._lbl_count = QLabel()
        self._lbl_count.setStyleSheet("color:#88aacc; font-size:12px;")
        ctrl.addWidget(self._lbl_count)
        ctrl.addStretch()

        self._btn_undo = QPushButton("⬅  Utolsó törlése")
        self._btn_undo.setToolTip(tr("Az utolsó pontpár vagy félkész görbe törlése"))
        self._btn_undo.clicked.connect(self._undo_last)
        self._btn_undo.setStyleSheet(
            "QPushButton{background:#3a2a1a; color:#ffbb88; padding:5px 14px;"
            "border-radius:4px; border:1px solid #664422;}"
            "QPushButton:hover{background:#4a3a2a;}"
            "QPushButton:disabled{background:#1e1e1e; color:#445;}")
        ctrl.addWidget(self._btn_undo)

        self._btn_reset = QPushButton("🗑  Mind törlése")
        self._btn_reset.setToolTip(tr("Összes pont törlése, újrakezdés"))
        self._btn_reset.clicked.connect(self._reset_all)
        self._btn_reset.setStyleSheet(
            "QPushButton{background:#3a1a1a; color:#ff8888; padding:5px 14px;"
            "border-radius:4px; border:1px solid #662222;}"
            "QPushButton:hover{background:#4a2a2a;}"
            "QPushButton:disabled{background:#1e1e1e; color:#445;}")
        ctrl.addWidget(self._btn_reset)

        self._btn_ok = QPushButton("✔  Igazítás számítása")
        self._btn_ok.setDefault(True)
        self._btn_ok.setToolTip(
            tr("Homográfia számítás RANSAC-kal a megjelölt GCP-párokból.\n"
               "Minimum 4 pár szükséges."))
        self._btn_ok.clicked.connect(self._accept_gcp)
        self._btn_ok.setStyleSheet(
            "QPushButton{background:#1a4a1a; color:#88ff88; font-weight:bold;"
            "padding:5px 18px; border-radius:4px; border:1px solid #228822;}"
            "QPushButton:hover{background:#205820;}"
            "QPushButton:disabled{background:#1e2430; color:#445;}")
        ctrl.addWidget(self._btn_ok)

        btn_cancel = QPushButton(tr("Mégse"))
        btn_cancel.clicked.connect(self.reject)
        btn_cancel.setStyleSheet(
            "QPushButton{background:#2a2a2a; color:#aaa; padding:5px 12px;"
            "border-radius:4px; border:1px solid #444;}"
            "QPushButton:hover{background:#363636;}")
        ctrl.addWidget(btn_cancel)

        root.addLayout(ctrl)

        # ── Segítség sor ─────────────────────────────────────────────────────
        help_lbl = QLabel(
            tr("<i>Görgetés: zoom  •  Jobb gomb+húzás: pásztázás  •  "
               "Bal gomb meglévő ponton+húzás: finomhangolás  •  "
               "Jobb klikk ponton (pont módban): pár törlése  •  "
               "Ív módban jobb klikk / Backspace: utolsó csúcs vissza  •  "
               "Esc: görbe megszakítása</i>"))
        help_lbl.setStyleSheet("color:#556677; font-size:11px;")
        help_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(help_lbl)

        # ── Szignál összekötések ──────────────────────────────────────────────
        self.canvas_a.point_clicked.connect(self._on_click_a)
        self.canvas_b.point_clicked.connect(self._on_click_b)
        self.canvas_a.point_moved.connect(self._on_moved_a)
        self.canvas_b.point_moved.connect(self._on_moved_b)
        self.canvas_a.point_deleted.connect(self._on_delete)
        self.canvas_b.point_deleted.connect(self._on_delete)
        self.canvas_a.curve_done.connect(self._on_curve_done_a)
        self.canvas_b.curve_done.connect(self._on_curve_done_b)

    # ── Mód-váltás ───────────────────────────────────────────────────────────

    _BTN_ON  = ("QPushButton{background:#1a3a4a; color:#22ddcc; font-weight:bold;"
                "padding:4px 14px; border-radius:4px; border:2px solid #22ddcc;}"
                "QPushButton:hover{background:#1e4a5a;}")
    _BTN_OFF = ("QPushButton{background:#2a2a3a; color:#8899aa; padding:4px 14px;"
                "border-radius:4px; border:1px solid #445;}"
                "QPushButton:hover{background:#363650;}")

    def _set_mode(self, mode: str) -> None:
        """Rajzolási mód váltása; félkész állapotok törlésével."""
        self._draw_mode       = mode
        self._pending_a       = None
        self._pending_curve_a = None
        self._next_side       = "A"
        self.canvas_a.set_draw_mode(mode)
        self.canvas_b.set_draw_mode(mode)
        self.canvas_a.cancel_curve()
        self.canvas_b.cancel_curve()
        # Mód-gombok vizuális kiemelése
        self._btn_mode_pt .setStyleSheet(self._BTN_ON  if mode == "point"    else self._BTN_OFF)
        self._btn_mode_pl .setStyleSheet(self._BTN_ON  if mode == "polyline" else self._BTN_OFF)
        self._btn_mode_arc.setStyleSheet(self._BTN_ON  if mode == "arc"      else self._BTN_OFF)
        # N-spinbox csak görbe módban releváns
        self._spin_n.setEnabled(mode != "point")
        self._refresh()

    # ── Pont mód – szignálok ─────────────────────────────────────────────────

    def _on_click_a(self, ix: float, iy: float) -> None:
        if self._draw_mode != "point" or self._next_side != "A":
            return
        self._pending_a = (ix, iy)
        self._next_side = "B"
        self._refresh()

    def _on_click_b(self, ix: float, iy: float) -> None:
        if self._draw_mode != "point" or self._next_side != "B" or self._pending_a is None:
            return
        self._pairs.append((self._pending_a, (ix, iy)))
        self._pending_a = None
        self._next_side = "A"
        self._refresh()

    # ── Görbe mód – szignálok ────────────────────────────────────────────────

    def _on_curve_done_a(self, pts: list) -> None:
        if self._draw_mode == "point":
            return
        self._pending_curve_a = pts
        self._next_side = "B"
        self._refresh()

    def _on_curve_done_b(self, pts: list) -> None:
        if self._draw_mode == "point" or self._pending_curve_a is None:
            return
        pa = self._pending_curve_a
        pb = pts
        self._pending_curve_a = None
        self._next_side = "A"

        n = self._spin_n.value()
        if self._draw_mode == "polyline":
            pts_a = PointEditorWidget._subdivide_polyline(pa, n)
            pts_b = PointEditorWidget._subdivide_polyline(pb, n)
        elif self._draw_mode == "arc" and len(pa) >= 3 and len(pb) >= 3:
            mid_a = pa[2 + (len(pa) - 2) // 2]
            mid_b = pb[2 + (len(pb) - 2) // 2]
            pts_a = PointEditorWidget._subdivide_arc(pa[0], mid_a, pa[1], n)
            pts_b = PointEditorWidget._subdivide_arc(pb[0], mid_b, pb[1], n)
        else:
            self._refresh()
            return

        for pa_pt, pb_pt in zip(pts_a, pts_b):
            self._pairs.append((tuple(pa_pt), tuple(pb_pt)))
        self._refresh()

    # ── Pont mozgatás / törlés ───────────────────────────────────────────────

    def _on_moved_a(self, idx: int, ix: float, iy: float) -> None:
        if 0 <= idx < len(self._pairs):
            self._pairs[idx] = ((ix, iy), self._pairs[idx][1])
            self._sync_points()

    def _on_moved_b(self, idx: int, ix: float, iy: float) -> None:
        if 0 <= idx < len(self._pairs):
            self._pairs[idx] = (self._pairs[idx][0], (ix, iy))
            self._sync_points()

    def _on_delete(self, idx: int) -> None:
        if 0 <= idx < len(self._pairs):
            self._pairs.pop(idx)
            self._refresh()

    # ── Undo / reset ─────────────────────────────────────────────────────────

    def _undo_last(self) -> None:
        if self._pending_curve_a is not None:
            # Félkész görbe A-n → visszavonás: vissza A-ra
            self._pending_curve_a = None
            self._next_side = "A"
            self.canvas_a.cancel_curve()
            self.canvas_b.cancel_curve()
        elif self._pending_a is not None:
            self._pending_a = None
            self._next_side = "A"
        elif self._pairs:
            self._pairs.pop()
        self._refresh()

    def _reset_all(self) -> None:
        self._pairs.clear()
        self._pending_a       = None
        self._pending_curve_a = None
        self._next_side       = "A"
        self.canvas_a.cancel_curve()
        self.canvas_b.cancel_curve()
        self._refresh()

    # ── Belső frissítés ──────────────────────────────────────────────────────

    def _sync_points(self) -> None:
        """Csak a pont-listákat frissíti (húzás közbeni gyors update)."""
        pts_a = [pa for pa, _ in self._pairs]
        pts_b = [pb for _, pb in self._pairs]
        self.canvas_a.set_points(pts_a)
        self.canvas_b.set_points(pts_b)
        n = len(self._pairs)
        self._btn_ok.setEnabled(n >= 4)
        self._lbl_count.setText(f"{n} pár  (ajánlott: 5–15)")

    def _refresh(self) -> None:
        """Teljes UI szinkronizáció az aktuális állapottal."""
        n        = len(self._pairs)
        a_active = (self._next_side == "A")

        # Canvas aktiválása / inaktiválása
        self.canvas_a.set_active(a_active)
        self.canvas_b.set_active(not a_active)
        # Az éppen inaktív canvas félkész görbéjét töröljük
        if a_active:
            self.canvas_b.cancel_curve()
        else:
            self.canvas_a.cancel_curve()

        # Pontok szinkronizálása
        pts_a = [pa for pa, _ in self._pairs]
        pts_b = [pb for _, pb in self._pairs]
        self.canvas_a.set_points(pts_a)
        self.canvas_b.set_points(pts_b)
        self.canvas_a.set_pending(self._pending_a)
        self.canvas_b.set_pending(None)

        # Instrukció szöveg
        if self._draw_mode == "point":
            if a_active:
                instr = (f"▶  Kattints az  A  képen egy jól azonosítható pontra  "
                         f"•  {n} pár megjelölve  •  minimum 4 szükséges")
            else:
                instr = "▶  Most kattints a  B  képen a megfelelő pontra  ◀"
        elif self._draw_mode == "polyline":
            if a_active:
                instr = (f"▶  Húzz vonalláncot az  A  képen  "
                         f"(bal+húzás = szakasz  •  2× klikk = kész)  •  {n} pár")
            else:
                instr = "▶  Most húzz megfelelő vonalláncot a  B  képen  ◀"
        else:  # arc
            if a_active:
                instr = (f"▶  Rajzolj ívet az  A  képen  "
                         f"(klikk: kezdőpont → végpont → köztes pontok → 2× klikk kész)"
                         f"  •  {n} pár")
            else:
                instr = "▶  Most rajzolj megfelelő ívet a  B  képen  ◀"

        self._lbl_instr.setText(instr)

        # Gombok
        has_pending = (self._pending_a is not None
                       or self._pending_curve_a is not None)
        self._btn_ok   .setEnabled(n >= 4)
        self._btn_undo .setEnabled(has_pending or n > 0)
        self._btn_reset.setEnabled(has_pending or n > 0)
        self._lbl_count.setText(f"{n} pár  (ajánlott: 5–15)")

    def _accept_gcp(self) -> None:
        if len(self._pairs) < 4:
            QMessageBox.warning(
                self,
                tr("Kevés pont"),
                tr("Legalább 4 pontpár szükséges a homográfia számításához."))
            return
        self.accept()

    # ── Publikus API ─────────────────────────────────────────────────────────

    def get_pairs(
        self,
    ) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Visszaadja a megjelölt GCP-párokat::

            [(pt_a, pt_b), ...]
            pt = (x_float, y_float)  képkoordinátában
        """
        return list(self._pairs)
