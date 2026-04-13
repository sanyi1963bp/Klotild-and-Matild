"""
theme_editor.py  –  ArchMorph Professional interaktív témaerkesztő

Ctrl+T / 🎨 gomb → be-/kikapcsolja a pipetta üzemmódot.

Üzemmódban:
  • Bal klikk bármely widgeten (képvászon kivételével)
        → elmenti az aktuális override-állapotot (undo-snapshot)
        → státuszsor mutatja a widget nevét és szelekorát
  • Jobb klikk bármely widgeten
        → helyi menü:
              – widget neve (nem kattintható fejléc)
              – a widgethez releváns összes szín-tulajdonság, mellettük
                egy színes téglalap (a jelenlegi szín)
              – rákattintva QColorDialog → azonnali alkalmazás
              ─────────────────────────────
              – ↩ Visszavonás  ← az utolsó bal-klikk snapshot-ra tér vissza
              – 💾 Séma mentése…
              – 📂 Séma betöltése  (almenü)
              – 🗑 Séma törlése    (almenü)
              – ↺ Visszaállítás alapértelmezettre
  • Esc → kilépés az üzemmódból

Képvásznakat (objectName vagy osztálynév tartalmaz 'canvas' szót) kihagyjuk.
QMenu-n, QMenuBar-on és toolbar gombokon a bal klikk NEM indítja el az
undo-snapshotot, hogy elkerüljük a végtelen ciklust.

Mentési hely: archmorph_themes.json  (a szkript könyvtárában)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PyQt6.QtCore    import QEvent, QObject, QPoint, QRect, Qt, QTimer, pyqtSignal
from PyQt6.QtGui     import QColor, QCursor, QIcon, QPainter, QPalette, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QColorDialog, QInputDialog, QMainWindow,
    QMenu, QMenuBar, QRubberBand, QToolBar, QToolButton,
)


# ── Típus-aliasok ──────────────────────────────────────────────────────────────
_Overrides = Dict[str, Dict[str, str]]   # {selector: {css-prop: hex}}
_Schemes   = Dict[str, _Overrides]       # {name: overrides}
_PropEntry = Tuple[str, str, QPalette.ColorRole]   # (css-prop, megjelenítési_név, palette_role)

# Widgetek, amelyeken bal klikken NEM snapshot-olunk (végtelen menü-ciklus elkerülése)
_SKIP_CLASSES = (QMenu, QMenuBar, QToolBar, QToolButton)


# ── Összes lehetséges QSS szín-tulajdonság ─────────────────────────────────────
#  (CSS tulajdonság, magyar megjelenítési név, QPalette fallback-szerep)
_ALL_COLOR_PROPS: List[_PropEntry] = [
    ("background-color",           "Háttér",                QPalette.ColorRole.Window),
    ("color",                      "Szöveg",                QPalette.ColorRole.WindowText),
    ("border-color",               "Keret",                 QPalette.ColorRole.Mid),
    ("selection-background-color", "Kijelölés háttere",     QPalette.ColorRole.Highlight),
    ("selection-color",            "Kijelölt szöveg",       QPalette.ColorRole.HighlightedText),
    ("alternate-background-color", "Váltakozó sor háttere", QPalette.ColorRole.AlternateBase),
    ("gridline-color",             "Rácsvonal",             QPalette.ColorRole.Mid),
    ("outline-color",              "Körvonal (fókusz)",     QPalette.ColorRole.Highlight),
]
_PROP_MAP: Dict[str, Tuple[str, QPalette.ColorRole]] = {
    p: (name, role) for p, name, role in _ALL_COLOR_PROPS
}

# ── Widget-típusonként releváns CSS szín-tulajdonságok ─────────────────────────
_base     = ["background-color", "color"]
_input    = _base + ["border-color", "selection-background-color", "selection-color"]
_listlike = _input + ["alternate-background-color"]
_frame    = _base + ["border-color"]

_WIDGET_PROPS: Dict[str, List[str]] = {
    # Alap konténerek
    "QWidget":          _base,
    "QMainWindow":      _base,
    "QDialog":          _frame,
    "QFrame":           _frame,
    "QGroupBox":        _frame,
    "QScrollArea":      ["background-color", "border-color"],
    "QSplitter":        ["background-color"],
    "QSplitterHandle":  ["background-color"],
    "QDockWidget":      _frame,
    # Szöveg / megjelenítés
    "QLabel":           _base,
    "QTextBrowser":     _input,
    # Beviteli mezők
    "QLineEdit":        _input,
    "QTextEdit":        _input,
    "QPlainTextEdit":   _input,
    "QSpinBox":         _input,
    "QDoubleSpinBox":   _input,
    "QDateEdit":        _input,
    "QTimeEdit":        _input,
    "QDateTimeEdit":    _input,
    "QComboBox":        _input,
    # Gombok
    "QPushButton":      _frame,
    "QToolButton":      _frame,
    "QCheckBox":        _base,
    "QRadioButton":     _base,
    # Listák / fák / táblák
    "QListWidget":      _listlike,
    "QListView":        _listlike,
    "QTreeWidget":      _listlike,
    "QTreeView":        _listlike,
    "QTableWidget":     _listlike + ["gridline-color"],
    "QTableView":       _listlike + ["gridline-color"],
    "QHeaderView":      _frame,
    # Fülek / eszköztár / menü / státusz
    "QTabWidget":       _frame,
    "QTabBar":          _frame,
    "QMenuBar":         _base + ["selection-background-color", "selection-color"],
    "QMenu":            _frame + ["selection-background-color", "selection-color"],
    "QToolBar":         ["background-color", "border-color"],
    "QStatusBar":       _base,
    # Egyebek
    "QScrollBar":       ["background-color", "border-color"],
    "QProgressBar":     _frame,
    "QSlider":          ["background-color", "border-color"],
    "QDial":            ["background-color", "border-color"],
    "QAbstractScrollArea": ["background-color", "border-color"],
}


class ThemeEditor(QObject):
    """
    Interaktív témaerkesztő motor.

    Paraméterek
    -----------
    app        : a futó QApplication példány
    window     : a MainWindow (státuszsor, parent dialog)
    base_qss   : az eredeti alkalmazás-stylesheet
    config_dir : a könyvtár ahol az archmorph_themes.json tárolódik
    """

    theme_changed = pyqtSignal()

    _SCHEMES_FILE = "archmorph_themes.json"

    def __init__(
        self,
        app:        QApplication,
        window:     QMainWindow,
        base_qss:   str,
        config_dir: Path,
    ) -> None:
        super().__init__(window)
        self._app        = app
        self._window     = window
        self._base_qss   = base_qss
        self._config_dir = config_dir
        self._active     = False
        self._paused     = False   # True: modális dialógus alatt szünetel a filter

        self._overrides:     _Overrides = {}
        self._schemes:       _Schemes   = {}
        self._active_scheme: str        = ""
        self._undo_snapshot: _Overrides = {}

        # Hover kiemelés
        self._rubber:         Optional[QRubberBand] = None
        self._hovered_widget: Optional[object]      = None

        self._load_from_file()
        if self._active_scheme and self._active_scheme in self._schemes:
            self._overrides = {s: dict(p)
                               for s, p in self._schemes[self._active_scheme].items()}
            self._apply_overrides()

    # ══════════════════════════════════════════════════════════════════════════
    # Be- / kikapcsolás
    # ══════════════════════════════════════════════════════════════════════════

    def toggle(self) -> None:
        if self._active:
            self._deactivate()
        else:
            self._activate()

    def _activate(self) -> None:
        self._active = True
        self._app.installEventFilter(self)
        QTimer.singleShot(50, lambda: (
            QApplication.setOverrideCursor(Qt.CursorShape.CrossCursor)
            if self._active else None
        ))
        self._window.statusBar().showMessage(
            "🎨  Témaerkesztő  –  "
            "Bal klikk: elem kijelölése  |  "
            "Jobb klikk: szín módosítása  |  "
            "Esc: kilépés"
        )

    def _deactivate(self) -> None:
        self._active = False
        self._app.removeEventFilter(self)
        QApplication.restoreOverrideCursor()
        self._clear_rubber()
        self._hovered_widget = None
        self._window.statusBar().showMessage("Témaerkesztő kikapcsolva.")

    @property
    def is_active(self) -> bool:
        return self._active

    # ══════════════════════════════════════════════════════════════════════════
    # Képvászon kizárása
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _is_image_canvas(widget) -> bool:
        """True, ha a widget egy képnézegető/szerkesztő vászon – kihagyjuk."""
        name = widget.objectName().lower()
        cls  = widget.__class__.__name__.lower()
        return "canvas" in name or "canvas" in cls

    # ══════════════════════════════════════════════════════════════════════════
    # Widget-típushoz tartozó szín-tulajdonságok
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _props_for_widget(widget) -> List[_PropEntry]:
        """
        Visszaadja a widgethez releváns CSS szín-tulajdonságok listáját.
        Az MRO-ban az első ismert osztályt keresi.
        """
        for klass in type(widget).__mro__:
            keys = _WIDGET_PROPS.get(klass.__name__)
            if keys is not None:
                break
        else:
            keys = _base

        return [
            (p, _PROP_MAP[p][0], _PROP_MAP[p][1])
            for p in keys
            if p in _PROP_MAP
        ]

    # ══════════════════════════════════════════════════════════════════════════
    # Event filter
    # ══════════════════════════════════════════════════════════════════════════

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # type: ignore[override]
        if not self._active or self._paused:
            return False

        t = event.type()

        # ── Esc → kilépés ─────────────────────────────────────────────────
        if t == QEvent.Type.KeyPress and hasattr(event, "key"):
            if event.key() == Qt.Key.Key_Escape:
                self._deactivate()
                return True

        # ── Hover → gumiszalag (képvásznon nem) ──────────────────────────
        if t == QEvent.Type.MouseMove:
            w = QApplication.widgetAt(QCursor.pos())
            if w is not None and w is not self._hovered_widget:
                self._hovered_widget = w
                if self._is_image_canvas(w):
                    self._clear_rubber()
                else:
                    self._update_rubber(w)
            return False

        # ── Kattintás ─────────────────────────────────────────────────────
        if t == QEvent.Type.MouseButtonPress and hasattr(event, "button"):
            w = QApplication.widgetAt(QCursor.pos())

            # Menü-elemek és képvásznok: érintetlenül hagyjuk
            if w is None:
                return False
            if isinstance(w, _SKIP_CLASSES) or self._is_image_canvas(w):
                return False

            btn = event.button()

            if btn == Qt.MouseButton.LeftButton:
                self._take_snapshot(w)
                return True

            if btn == Qt.MouseButton.RightButton:
                self._show_color_menu(w, QCursor.pos())
                return True

        return False

    # ══════════════════════════════════════════════════════════════════════════
    # Hover kiemelés
    # ══════════════════════════════════════════════════════════════════════════

    def _update_rubber(self, widget) -> None:
        if self._rubber is None:
            self._rubber = QRubberBand(QRubberBand.Shape.Rectangle)
        try:
            gp = widget.mapToGlobal(QPoint(0, 0))
            self._rubber.setGeometry(QRect(gp, widget.size()))
            self._rubber.show()
        except RuntimeError:
            self._rubber = None

    def _clear_rubber(self) -> None:
        if self._rubber is not None:
            self._rubber.hide()

    # ══════════════════════════════════════════════════════════════════════════
    # Bal klikk: snapshot
    # ══════════════════════════════════════════════════════════════════════════

    def _take_snapshot(self, widget) -> None:
        self._undo_snapshot = {s: dict(p) for s, p in self._overrides.items()}
        selector = self._selector_for(widget)
        props    = self._props_for_widget(widget)
        prop_str = ", ".join(name for _, name, _ in props)
        self._window.statusBar().showMessage(
            f"Kijelölve: {widget.__class__.__name__}  [{selector}]"
            f"   –   Elérhető tulajdonságok: {prop_str}"
            "   |   Jobb klikk: szín módosítása  |  Esc: kilépés"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Jobb klikk: szín-menü
    # ══════════════════════════════════════════════════════════════════════════

    def _show_color_menu(self, widget, pos: QPoint) -> None:
        selector = self._selector_for(widget)
        cur_ov   = self._overrides.get(selector, {})
        props    = self._props_for_widget(widget)

        menu = QMenu(self._window)

        # ── Fejléc ────────────────────────────────────────────────────────
        hdr = menu.addAction(f"  {widget.__class__.__name__}   [{selector}]")
        hdr.setEnabled(False)
        menu.addSeparator()

        # ── Szín-sorok ────────────────────────────────────────────────────
        for prop, label, palette_role in props:
            hex_color = cur_ov.get(prop, self._palette_color(widget, palette_role))
            icon = self._color_icon(hex_color)
            act  = menu.addAction(icon, f"  {label}   {hex_color}")
            act.triggered.connect(
                lambda _=False, w=widget, p=prop:
                    self._pick_and_apply(w, p))

        menu.addSeparator()

        # ── Visszavonás ───────────────────────────────────────────────────
        undo_act = menu.addAction("↩  Visszavonás")
        undo_act.setToolTip("Visszaáll az utolsó bal-klikk előtti állapotra")
        undo_act.triggered.connect(self._undo)

        menu.addSeparator()

        # ── Séma-kezelés ──────────────────────────────────────────────────
        save_act = menu.addAction("💾  Séma mentése…")
        save_act.triggered.connect(self._save_scheme_dialog)

        load_menu = menu.addMenu("📂  Séma betöltése")
        if self._schemes:
            for name in sorted(self._schemes):
                act = load_menu.addAction(name)
                act.setCheckable(True)
                act.setChecked(name == self._active_scheme)
                act.triggered.connect(
                    lambda _=False, n=name: self._load_scheme(n))
        else:
            no = load_menu.addAction("(Nincs mentett séma)")
            no.setEnabled(False)

        if self._schemes:
            del_menu = menu.addMenu("🗑  Séma törlése")
            for name in sorted(self._schemes):
                act = del_menu.addAction(name)
                act.triggered.connect(
                    lambda _=False, n=name: self._delete_scheme(n))

        menu.addSeparator()

        reset_act = menu.addAction("↺  Visszaállítás alapértelmezettre")
        reset_act.setToolTip("Minden egyéni szín törlése – az eredeti sötét téma visszaáll.")
        reset_act.triggered.connect(self._reset_to_default)

        # Menü bezárása után visszaállítjuk a pipetta kurzort
        # (de csak ha nincs modális dialógus nyitva)
        menu.aboutToHide.connect(
            lambda: QTimer.singleShot(
                10,
                lambda: QApplication.setOverrideCursor(
                    Qt.CursorShape.CrossCursor
                ) if self._active and not self._paused else None
            )
        )

        QApplication.restoreOverrideCursor()
        menu.exec(pos)

    # ══════════════════════════════════════════════════════════════════════════
    # Szín-ikon (kis téglalap)
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _color_icon(hex_color: str) -> QIcon:
        pix = QPixmap(24, 14)
        try:
            fill = QColor(hex_color)
            if not fill.isValid():
                fill = QColor("#888888")
        except Exception:
            fill = QColor("#888888")
        pix.fill(fill)
        p = QPainter(pix)
        p.setPen(QColor("#555555"))
        p.drawRect(0, 0, 23, 13)
        p.end()
        return QIcon(pix)

    # ══════════════════════════════════════════════════════════════════════════
    # Paletta-szín kiolvasása (ha nincs override)
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _palette_color(widget, role: QPalette.ColorRole) -> str:
        try:
            color = widget.palette().color(role)
            if color.isValid():
                return color.name()
        except Exception:
            pass
        return "#888888"

    # ══════════════════════════════════════════════════════════════════════════
    # Szín kiválasztása és azonnali alkalmazás
    # ══════════════════════════════════════════════════════════════════════════

    def _pick_and_apply(self, widget, prop: str) -> None:
        selector = self._selector_for(widget)
        _, palette_role = _PROP_MAP.get(
            prop, ("", QPalette.ColorRole.Window))
        cur_hex = self._overrides.get(selector, {}).get(
            prop, self._palette_color(widget, palette_role))

        # ── Modális dialógus előtt: szüneteltetjük a filtert ──────────────
        self._paused = True
        QApplication.restoreOverrideCursor()

        color = QColorDialog.getColor(
            QColor(cur_hex), self._window,
            f"{widget.__class__.__name__}  –  {prop}")

        # ── Dialógus után: felébredünk ────────────────────────────────────
        self._paused = False
        if self._active:
            QTimer.singleShot(
                50,
                lambda: QApplication.setOverrideCursor(
                    Qt.CursorShape.CrossCursor) if self._active else None)

        if color.isValid():
            self._overrides.setdefault(selector, {})[prop] = color.name()
            self._apply_overrides()
            self.theme_changed.emit()

    # ══════════════════════════════════════════════════════════════════════════
    # Visszavonás
    # ══════════════════════════════════════════════════════════════════════════

    def _undo(self) -> None:
        self._overrides = {s: dict(p) for s, p in self._undo_snapshot.items()}
        self._apply_overrides()
        self.theme_changed.emit()
        self._window.statusBar().showMessage("Visszavonva – előző állapot visszaállítva.")

    # ══════════════════════════════════════════════════════════════════════════
    # QSS alkalmazás
    # ══════════════════════════════════════════════════════════════════════════

    def _apply_overrides(self) -> None:
        lines = [
            f"{sel} {{ {'; '.join(f'{k}: {v}' for k, v in props.items())} }}"
            for sel, props in self._overrides.items()
            if props
        ]
        extra = "\n".join(lines)
        self._app.setStyleSheet(
            self._base_qss + ("\n" + extra if extra else ""))

    # ══════════════════════════════════════════════════════════════════════════
    # Séma-kezelés
    # ══════════════════════════════════════════════════════════════════════════

    def _save_scheme_dialog(self) -> None:
        self._paused = True
        QApplication.restoreOverrideCursor()

        name, ok = QInputDialog.getText(
            self._window, "Séma mentése",
            "Add meg a színséma nevét:",
            text=self._active_scheme)

        self._paused = False
        if self._active:
            QTimer.singleShot(
                50,
                lambda: QApplication.setOverrideCursor(
                    Qt.CursorShape.CrossCursor) if self._active else None)

        if not (ok and name.strip()):
            return
        name = name.strip()
        self._schemes[name] = {s: dict(p) for s, p in self._overrides.items()}
        self._active_scheme  = name
        self._save_to_file()
        self._window.statusBar().showMessage(f"Színséma mentve: '{name}'")

    def _load_scheme(self, name: str) -> None:
        if name not in self._schemes:
            return
        self._overrides     = {s: dict(p)
                               for s, p in self._schemes[name].items()}
        self._active_scheme = name
        self._apply_overrides()
        self.theme_changed.emit()
        self._window.statusBar().showMessage(f"Színséma betöltve: '{name}'")

    def _delete_scheme(self, name: str) -> None:
        if name not in self._schemes:
            return
        del self._schemes[name]
        if self._active_scheme == name:
            self._active_scheme = ""
        self._save_to_file()
        self._window.statusBar().showMessage(f"Színséma törölve: '{name}'")

    def _reset_to_default(self) -> None:
        self._overrides     = {}
        self._active_scheme = ""
        self._app.setStyleSheet(self._base_qss)
        self.theme_changed.emit()
        self._window.statusBar().showMessage("Téma visszaállítva alapértelmezettre.")

    # Kontextusmenü közvetlenül a toolbar gombról (üzemmódon kívül is)
    def show_context_menu(self, pos: QPoint) -> None:
        menu = QMenu(self._window)

        save_act = menu.addAction("💾  Séma mentése…")
        save_act.triggered.connect(self._save_scheme_dialog)

        load_menu = menu.addMenu("📂  Séma betöltése")
        if self._schemes:
            for name in sorted(self._schemes):
                act = load_menu.addAction(name)
                act.setCheckable(True)
                act.setChecked(name == self._active_scheme)
                act.triggered.connect(
                    lambda _=False, n=name: self._load_scheme(n))
        else:
            no = load_menu.addAction("(Nincs mentett séma)")
            no.setEnabled(False)

        if self._schemes:
            del_menu = menu.addMenu("🗑  Séma törlése")
            for name in sorted(self._schemes):
                act = del_menu.addAction(name)
                act.triggered.connect(
                    lambda _=False, n=name: self._delete_scheme(n))

        menu.addSeparator()
        reset_act = menu.addAction("↺  Visszaállítás alapértelmezettre")
        reset_act.triggered.connect(self._reset_to_default)
        menu.exec(pos)

    # ══════════════════════════════════════════════════════════════════════════
    # Fájl I/O
    # ══════════════════════════════════════════════════════════════════════════

    def _schemes_path(self) -> Path:
        return self._config_dir / self._SCHEMES_FILE

    def _save_to_file(self) -> None:
        try:
            data = {"active_scheme": self._active_scheme,
                    "schemes":       self._schemes}
            p = self._schemes_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            self._window.statusBar().showMessage(f"Séma mentési hiba: {exc}")

    def _load_from_file(self) -> None:
        try:
            p = self._schemes_path()
            if p.exists():
                with p.open(encoding="utf-8") as f:
                    data = json.load(f)
                self._schemes       = data.get("schemes", {})
                self._active_scheme = data.get("active_scheme", "")
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════════════════
    # Segéd
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _selector_for(widget) -> str:
        name = widget.objectName()
        if name:
            return f"#{name}"
        return widget.__class__.__name__
