"""
config_editor.py  –  ArchMorph Professional – Vizuális konfigurációs szerkesztő
================================================================================
Grafikus felület az archmorph_config.toml szerkesztéséhez.

Funkciók
--------
  • Minden TOML beállítás widgeten át szerkeszthető
  • Színek     – kattintásra QColorDialog; jobb klikk: másolás / reset
  • RGBA fill  – 4 csúszka (R/G/B/A) + élő előnézet; jobb klikk: reset
  • Számok     – QSpinBox / QDoubleSpinBox; jobb klikk: reset alapértelmezetthez
  • Bool       – QCheckBox
  • Választék  – QComboBox
  • Profil     – mentés / betöltés / törlés névvel ellátott .toml fájlokba
                 (profiles/ alkönyvtár a konfig fájl mellett)
  • Ctrl+S     – gyors mentés

Megjegyzés
----------
Színváltozások (module-level QColor konstansok) csak újraindítás után lépnek életbe.
Algoritmus-paraméterek következő illesztési művelettől érvényesek.
"""
from __future__ import annotations

import sys
import pathlib
from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QShortcut, QKeySequence
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox,
    QFrame, QGroupBox, QHBoxLayout, QLabel, QMenu, QMessageBox,
    QPushButton, QScrollArea, QSizePolicy, QSlider, QSpinBox,
    QVBoxLayout, QWidget,
)

# ── Config loader ─────────────────────────────────────────────────────────────
try:
    from archmorph_config_loader import cfg, reload as cfg_reload
    import archmorph_config_loader as _cfg_mod
    _CONFIG_FILE: pathlib.Path = _cfg_mod._CONFIG_FILE
except ImportError:
    def cfg(k, d): return d                          # type: ignore[misc]
    def cfg_reload(): pass                           # type: ignore[misc]
    _CONFIG_FILE = pathlib.Path(__file__).parent / "archmorph_config.toml"


try:
    from TRANSLATIONS import tr
except ImportError:
    def tr(text: str) -> str: return text

_PROFILES_DIR: pathlib.Path = _CONFIG_FILE.parent / "profiles"


# ══════════════════════════════════════════════════════════════════════════════
#  SCHEMA – minden beállítás leírása
# ══════════════════════════════════════════════════════════════════════════════
# Minden elem:
#   key      – TOML kulcs (a prefix-hez viszonyítva)
#   label    – megjelenítendő felirat
#   type     – "color" | "rgba" | "int" | "float" | "bool" | "choice"
#   default  – beépített alapértelmezett érték
#   desc     – rövid magyarázat (opcionális)
#   range    – (min, max) int/float esetén
#   decimals – tizedes helyek float esetén
#   step     – lépésköz float esetén
#   choices  – choice típusnál a lehetséges értékek listája

_SCHEMA: List[Dict] = [
    {
        "section": tr("🌐  Nyelv / Language"),
        "prefix": "ui",
        "keys": [
            {"key": "language", "label": tr("Felület nyelve"), "type": "choice",
             "default": "hu", "choices": ["hu", "en"],
             "desc": tr("A program felhasználói felületének nyelve. Újraindítás szükséges.")},
        ],
    },
    # ── Pontok – körvonal ─────────────────────────────────────────────────────
    {
        "section": tr("🎨  Pontok – Körvonal"),
        "prefix": "ui.colors.points",
        "keys": [
            {"key": "normal",      "label": tr("Normál"),          "type": "color",
             "default": "#ff8a3d",
             "desc": tr("Nem kijelölt pont körének vonalszíne.")},
            {"key": "active",      "label": tr("Aktív"),           "type": "color",
             "default": "#ff3333",
             "desc": tr("Éppen kiválasztott pont (kereszthajszal jelenik meg).")},
            {"key": "multi",       "label": tr("Multi-kijelölt"),  "type": "color",
             "default": "#44aaff",
             "desc": tr("Gumiszalag-kerettel egyszerre kijelölt pontok.")},
            {"key": "text",        "label": tr("Sorszám szín"),    "type": "color",
             "default": "#ffffff",
             "desc": tr("Pontok mellé rajzolt sorszám felirat normál állapotban.")},
            {"key": "text_active", "label": tr("Sorszám (aktív)"), "type": "color",
             "default": "#ffcccc",
             "desc": tr("Sorszám felirat az aktív pont mellett.")},
        ],
    },
    # ── Pontok – kitöltés ─────────────────────────────────────────────────────
    {
        "section": tr("🎨  Pontok – Belső kitöltés (RGBA)"),
        "prefix": "ui.colors.points.fill",
        "keys": [
            {"key": "normal_rgba", "label": tr("Normál fill"),     "type": "rgba",
             "default": (255, 138,  61,  70),
             "desc": tr("Nem kijelölt pont belső kitöltése. A = átlátszatlanság (0–255).")},
            {"key": "active_rgba", "label": tr("Aktív fill"),      "type": "rgba",
             "default": (255,  51,  51, 150),
             "desc": tr("Aktív pont belső kitöltése.")},
            {"key": "multi_rgba",  "label": tr("Multi fill"),      "type": "rgba",
             "default": ( 68, 170, 255,  90),
             "desc": tr("Multi-kijelölt pont belső kitöltése.")},
        ],
    },
    # ── ROI ───────────────────────────────────────────────────────────────────
    {
        "section": tr("🟠  ROI – Keresési ablak"),
        "prefix": "ui.colors.roi",
        "keys": [
            {"key": "border",       "label": tr("Keret szín"),   "type": "color",
             "default": "#ff9900",
             "desc": tr("ROI téglalap keretének vonalszíne. Ctrl+húzással rajzolható.")},
            {"key": "handle",       "label": tr("Fogópont"),     "type": "color",
             "default": "#ffffff",
             "desc": tr("A 8 átméretező fogópont kitöltési színe.")},
            {"key": "fill_rgba",    "label": tr("Belső fill"),   "type": "rgba",
             "default": (255, 153, 0, 22),
             "desc": tr("ROI belső áttetsző kitöltése.")},
            {"key": "outside_rgba", "label": tr("Külső árnyék"), "type": "rgba",
             "default": (0, 0, 0, 90),
             "desc": tr("ROI-n kívüli terület elsötétítő réteg.")},
        ],
    },
    # ── Vászon ────────────────────────────────────────────────────────────────
    {
        "section": tr("🖼  Vászon háttér"),
        "prefix": "ui.colors.canvas",
        "keys": [
            {"key": "background",  "label": tr("Háttér"),        "type": "color",
             "default": "#1a1f24",
             "desc": tr("Vászon háttérszíne (kép mögötti terület).")},
            {"key": "border",      "label": tr("Keret"),         "type": "color",
             "default": "#343b45",
             "desc": tr("Vászon keretes kerete.")},
            {"key": "placeholder", "label": tr("Helytartó"),     "type": "color",
             "default": "#4a5260",
             "desc": tr("'Nincs kép betöltve' szöveg és vázlat szín.")},
        ],
    },
    # ── Gumiszalag ────────────────────────────────────────────────────────────
    {
        "section": tr("⬜  Gumiszalag-keret (húzásos kijelölés)"),
        "prefix": "ui.colors.rband",
        "keys": [
            {"key": "selection", "label": tr("Kijelölés keret"), "type": "color",
             "default": "#88bbff",
             "desc": tr("Bal gomb + húzás: pontokat kijelölő szaggatott keret.")},
            {"key": "roi",       "label": tr("ROI keret"),       "type": "color",
             "default": "#ffcc44",
             "desc": tr("Ctrl + húzás: ROI-t rajzoló szaggatott keret.")},
        ],
    },
    # ── Sokszög-ROI ───────────────────────────────────────────────────────────
    {
        "section": tr("🔷  Sokszög-ROI"),
        "prefix": "ui.colors.polygon",
        "keys": [
            {"key": "draw",      "label": tr("Rajzolás"),        "type": "color",
             "default": "#FFB300",
             "desc": tr("Sokszög rajzolás közben (még nem lezárt).")},
            {"key": "done",      "label": tr("Lezárt"),          "type": "color",
             "default": "#FF6F00",
             "desc": tr("Lezárt sokszög (helyi menü aktív állapot).")},
            {"key": "fill_rgba", "label": tr("Kitöltés"),        "type": "rgba",
             "default": (255, 179, 0, 35),
             "desc": tr("Sokszög belsejének áttetsző kitöltése.")},
        ],
    },
    # ── Export előnézet ───────────────────────────────────────────────────────
    {
        "section": tr("🎬  Export előnézet"),
        "prefix": "ui.colors.preview",
        "keys": [
            {"key": "background", "label": tr("Háttér"),         "type": "color",
             "default": "#0d1117",
             "desc": tr("Az export előnézet vászon háttérszíne.")},
        ],
    },
    # ── Téma ──────────────────────────────────────────────────────────────────
    {
        "section": tr("🎨  UI Téma – Globális sötét téma"),
        "prefix": "ui.colors.theme",
        "keys": [
            {"key": "window_bg",    "label": tr("Ablak háttér"),      "type": "color",
             "default": "#1a1f24",  "desc": tr("Főablak háttérszíne.")},
            {"key": "panel_bg",     "label": tr("Panel háttér"),      "type": "color",
             "default": "#252a33",  "desc": tr("Panelek, fülek, beviteli mezők háttere.")},
            {"key": "statusbar_bg", "label": tr("Állapotsor"),        "type": "color",
             "default": "#111418",  "desc": tr("Állapotsor + eszköztár háttere.")},
            {"key": "button_bg",    "label": tr("Gomb"),              "type": "color",
             "default": "#2c3340",  "desc": tr("Gombok alap háttere.")},
            {"key": "button_hover", "label": tr("Gomb (hover)"),      "type": "color",
             "default": "#3a4252",  "desc": tr("Gombok hover állapota.")},
            {"key": "input_bg",     "label": tr("Beviteli mező"),     "type": "color",
             "default": "#252a33",  "desc": tr("QSpinBox, QComboBox, QLineEdit háttere.")},
            {"key": "separator",    "label": tr("Elválasztó"),        "type": "color",
             "default": "#343b45",  "desc": tr("Elválasztó vonalak, csoportdoboz-keretek.")},
            {"key": "text_main",    "label": tr("Fő szöveg"),         "type": "color",
             "default": "#d7dde5",  "desc": tr("Elsődleges szövegszín.")},
            {"key": "text_dim",     "label": tr("Halvány szöveg"),    "type": "color",
             "default": "#888888",  "desc": tr("Halkított szöveg (állapotsor, kis feliratok).")},
            {"key": "text_label",   "label": tr("Felirat"),           "type": "color",
             "default": "#aaaaaa",  "desc": tr("QLabel feliratok.")},
            {"key": "group_title",  "label": tr("Csoport fejléc"),    "type": "color",
             "default": "#99aaaa",  "desc": tr("QGroupBox fejléc szöveg.")},
            {"key": "stale_bg",     "label": tr("Elavult gomb h."),   "type": "color",
             "default": "#5a3800",  "desc": tr("'Újragenerálás szükséges' gomb háttere.")},
            {"key": "stale_border", "label": tr("Elavult gomb k."),   "type": "color",
             "default": "#e8a020",  "desc": tr("'Újragenerálás szükséges' gomb kerete.")},
            {"key": "stale_text",   "label": tr("Elavult gomb sz."),  "type": "color",
             "default": "#ffcc66",  "desc": tr("'Újragenerálás szükséges' gomb szövege.")},
        ],
    },
    # ── Méretek ───────────────────────────────────────────────────────────────
    {
        "section": tr("📐  Méretek (pixel)"),
        "prefix": "ui.sizes",
        "keys": [
            {"key": "point_radius_normal", "label": tr("Pont sugár – normál"),     "type": "int",
             "default": 6,  "range": (1, 30),
             "desc": tr("Nem kijelölt pont körének sugara px-ben.")},
            {"key": "point_radius_active", "label": tr("Pont sugár – aktív"),      "type": "int",
             "default": 10, "range": (1, 30),
             "desc": tr("Aktív (kijelölt) pont körének sugara px-ben.")},
            {"key": "point_radius_multi",  "label": tr("Pont sugár – multi"),      "type": "int",
             "default": 8,  "range": (1, 30),
             "desc": tr("Multi-kijelölt pont körének sugara px-ben.")},
            {"key": "point_hit_radius",    "label": tr("Kattintási érzékenység"),  "type": "int",
             "default": 14, "range": (5, 50),
             "desc": tr("Ennél közelebb kattintva a pont kijelölődik. Nagyobb = könnyebb kattintani.")},
            {"key": "roi_handle_radius",   "label": tr("ROI fogópont sugara"),     "type": "int",
             "default": 6,  "range": (2, 20),
             "desc": tr("ROI átméretező fogópontok körének sugara px-ben.")},
            {"key": "roi_min_size",        "label": tr("Minimum ROI méret"),       "type": "int",
             "default": 10, "range": (5, 100),
             "desc": tr("Ennél kisebb Ctrl+húzás figyelmen kívül marad.")},
        ],
    },
    # ── SuperPoint ────────────────────────────────────────────────────────────
    {
        "section": tr("🔬  SuperPoint + LightGlue"),
        "prefix": "match.superpoint",
        "keys": [
            {"key": "max_keypoints",       "label": tr("Max. kulcspontok"),      "type": "int",
             "default": 4096, "range": (128, 8192),
             "desc": tr("Több → pontosabb, de lassabb és több GPU-memória. (cvg/LightGlue alapértelmezett: 4096)")},
            {"key": "detection_threshold", "label": tr("Detekciós küszöb"),      "type": "float",
             "default": 0.005, "range": (0.0001, 0.99), "decimals": 4, "step": 0.001,
             "desc": tr("Kisebb → több gyengébb pont; nagyobb → kevesebb, erősebb. (SuperPoint official: 0.005)")},
            {"key": "nms_radius",          "label": tr("NMS sugár (px)"),         "type": "int",
             "default": 4, "range": (1, 20),
             "desc": tr("Non-Maximum Suppression: pontok legalább ennyi px-re legyenek egymástól. (DeTone 2018: 4)")},
            {"key": "match_threshold",     "label": tr("Match küszöb (LG)"),      "type": "float",
             "default": 0.10, "range": (0.01, 1.0), "decimals": 3, "step": 0.01,
             "desc": tr("LightGlue szűrési küszöb – kisebb → szigorúbb. (LightGlue alapértelmezett: 0.1)")},
            {"key": "depth_confidence",    "label": tr("Mélység konfidencia"),    "type": "float",
             "default": 0.95, "range": (0.5, 1.0), "decimals": 2, "step": 0.01,
             "desc": tr("Korai leállás küszöb – ha megbízható → gyorsabb futás. (ICCV 2023: 0.95)")},
            {"key": "width_confidence",    "label": tr("Szélesség konfidencia"),  "type": "float",
             "default": 0.99, "range": (0.5, 1.0), "decimals": 2, "step": 0.01,
             "desc": tr("Korai leállás küszöb szélesség irányban. (ICCV 2023: 0.99)")},
        ],
    },
    # ── DISK ──────────────────────────────────────────────────────────────────
    {
        "section": tr("💿  DISK + LightGlue"),
        "prefix": "match.disk",
        "keys": [
            {"key": "max_keypoints",   "label": tr("Max. kulcspontok"),     "type": "int",
             "default": 5000, "range": (128, 8192),
             "desc": tr("DISK sűrűbben mintavételez – magasabb érték ajánlott. (DeDoDe/kornia: ~5000–10000)")},
            {"key": "match_threshold", "label": tr("Match küszöb (LG)"),    "type": "float",
             "default": 0.10, "range": (0.01, 1.0), "decimals": 3, "step": 0.01,
             "desc": tr("LightGlue szűrési küszöb. (LightGlue alapértelmezett: 0.1)")},
        ],
    },
    # ── LoFTR ─────────────────────────────────────────────────────────────────
    {
        "section": tr("🔭  LoFTR"),
        "prefix": "match.loftr",
        "keys": [
            {"key": "pretrained",     "label": tr("Modell típus"),          "type": "choice",
             "default": "outdoor", "choices": ["outdoor", "indoor"],
             "desc": tr("outdoor: épületek, terek, tájak  |  indoor: szobák, belső terek.")},
            {"key": "conf_threshold", "label": tr("Konfidencia küszöb"),    "type": "float",
             "default": 0.50, "range": (0.0, 1.0), "decimals": 2, "step": 0.05,
             "desc": tr("Csak ennél megbízhatóbb egyezések maradnak. (paper: 0.2; morfinghoz: 0.5)")},
        ],
    },
    # ── SIFT ──────────────────────────────────────────────────────────────────
    {
        "section": tr("🔍  SIFT"),
        "prefix": "match.sift",
        "keys": [
            {"key": "max_keypoints",      "label": tr("Max. kulcspontok"),   "type": "int",
             "default": 0, "range": (0, 8192),
             "desc": tr("0 = korlátlan (Lowe 2004 + OpenCV alapértelmezettje). Nagy képnél: 2000–5000.")},
            {"key": "octave_layers",      "label": tr("Oktáv-rétegek"),      "type": "int",
             "default": 3, "range": (1, 10),
             "desc": tr("Gauss-skálatér oktávonkénti rétegei. (Lowe 2004: s=3) – általában ne változtasd.")},
            {"key": "contrast_threshold", "label": tr("Kontraszt küszöb"),   "type": "float",
             "default": 0.04, "range": (0.001, 0.5), "decimals": 3, "step": 0.005,
             "desc": tr("Kisebb → több gyenge pont; nagyobb → csak erős. (Lowe 2004: 0.04)")},
            {"key": "edge_threshold",     "label": tr("Él küszöb"),          "type": "float",
             "default": 10.0, "range": (1.0, 50.0), "decimals": 1, "step": 1.0,
             "desc": tr("Él-jellegű instabil pontok szűrése. (Lowe 2004: r=10)")},
            {"key": "sigma",              "label": tr("Gauss sigma"),         "type": "float",
             "default": 1.6, "range": (0.5, 5.0), "decimals": 1, "step": 0.1,
             "desc": tr("Gauss-simítás sigma értéke. (Lowe 2004: σ=1.6) – általában ne változtasd.")},
            {"key": "ratio_threshold",    "label": tr("Lowe arány küszöb"),  "type": "float",
             "default": 0.80, "range": (0.5, 0.99), "decimals": 2, "step": 0.01,
             "desc": tr("Kisebb → szigorúbb; 0.80 = Lowe 2004 optimuma ('90% hamis egyezés kiszűrve').")},
        ],
    },
    # ── RANSAC ────────────────────────────────────────────────────────────────
    {
        "section": tr("🎯  RANSAC – Geometriai szűrő"),
        "prefix": "match.ransac",
        "keys": [
            {"key": "enabled",          "label": tr("Bekapcsolva"),       "type": "bool",
             "default": True,
             "desc": tr("Geometriai outlier-szűrő. Erősen ajánlott bekapcsolva hagyni.")},
            {"key": "reproj_threshold", "label": tr("Reproj. küszöb"),    "type": "float",
             "default": 3.0, "range": (0.5, 20.0), "decimals": 1, "step": 0.5,
             "desc": tr("Visszavetítési küszöb px-ben. Kisebb → szigorúbb. (OpenCV alapértelmezett: 3.0 px)")},
        ],
    },
    # ── Export ────────────────────────────────────────────────────────────────
    {
        "section": tr("📤  Export alapértelmezések"),
        "prefix": "export.defaults",
        "keys": [
            {"key": "frame_count", "label": tr("Képkockák száma"),   "type": "int",
             "default": 40, "range": (4, 300),
             "desc": tr("Több → simább animáció, de lassabb generálás és nagyobb fájl.")},
            {"key": "fps",         "label": tr("FPS"),               "type": "float",
             "default": 25.0, "range": (1.0, 60.0), "decimals": 2, "step": 1.0,
             "desc": tr("Lejátszási sebesség. MP4: 24/25/30. GIF: max ~50 fps.")},
            {"key": "method",      "label": tr("Morph módszer"),     "type": "choice",
             "default": "Delaunay háromszög",
             "choices": ["Delaunay háromszög", "Optikai folyam", "Homográfia"],
             "desc": tr("Alapértelmezett morph módszer az Export lapfülön.")},
            {"key": "easing",      "label": tr("Easing görbe"),      "type": "choice",
             "default": "S-görbe",
             "choices": ["Lineáris", "Lassú start", "Lassú vége", "S-görbe"],
             "desc": tr("Alapértelmezett animáció-sebességgörbe.")},
            {"key": "ping_pong",   "label": tr("Ping-pong"),         "type": "bool",
             "default": False,
             "desc": tr("A→B→A hurok: az animáció visszafelé is lejátszik.")},
        ],
    },
    # ── Farneback – Gyors ─────────────────────────────────────────────────────
    {
        "section": tr("💨  Optikai folyam – Gyors preset"),
        "prefix": "flow.presets.fast",
        "keys": [
            {"key": "pyr_scale",  "label": tr("Piramis arány"),    "type": "float",
             "default": 0.5, "range": (0.1, 0.9), "decimals": 1, "step": 0.1,
             "desc": tr("Szintenkénti méretcsökkentés aránya. 0.5 = klasszikus piramis.")},
            {"key": "levels",     "label": tr("Piramisszintek"),   "type": "int",
             "default": 3, "range": (1, 8),
             "desc": tr("Több szint → nagyobb elmozdulás is nyomozható.")},
            {"key": "winsize",    "label": tr("Ablakméret (px)"),  "type": "int",
             "default": 15, "range": (5, 63),
             "desc": tr("Átlagolási ablak. Nagyobb → robusztusabb, de elmosódottabb. (OpenCV tutorial: 15)")},
            {"key": "iterations", "label": tr("Iterációk"),        "type": "int",
             "default": 3, "range": (1, 10),
             "desc": tr("Iterációk száma szintenként. Több → pontosabb, lassabb.")},
            {"key": "poly_n",     "label": tr("Poly N"),           "type": "int",
             "default": 5, "range": (3, 9),
             "desc": tr("Polinomexpanzió szomszédsága. 5 = gyors; 7 = simább (Farneback 2003).")},
            {"key": "poly_sigma", "label": tr("Poly sigma"),       "type": "float",
             "default": 1.1, "range": (0.5, 3.0), "decimals": 1, "step": 0.1,
             "desc": tr("poly_n=5 → 1.1; poly_n=7 → 1.5 ajánlott (Farneback 2003).")},
        ],
    },
    # ── Farneback – Normál ────────────────────────────────────────────────────
    {
        "section": tr("💨  Optikai folyam – Normál preset"),
        "prefix": "flow.presets.normal",
        "keys": [
            {"key": "pyr_scale",  "label": tr("Piramis arány"),    "type": "float",
             "default": 0.5, "range": (0.1, 0.9), "decimals": 1, "step": 0.1,
             "desc": tr("Szintenkénti méretcsökkentés aránya. 0.5 = klasszikus piramis.")},
            {"key": "levels",     "label": tr("Piramisszintek"),   "type": "int",
             "default": 4, "range": (1, 8),
             "desc": tr("Több szint → nagyobb elmozdulás is nyomozható. Normál: 4.")},
            {"key": "winsize",    "label": tr("Ablakméret (px)"),  "type": "int",
             "default": 21, "range": (5, 63),
             "desc": tr("Átlagolási ablak mérete. Nagyobb → robusztusabb, de elmosódottabb.")},
            {"key": "iterations", "label": tr("Iterációk"),        "type": "int",
             "default": 4, "range": (1, 10),
             "desc": tr("Iterációk száma szintenként. Több → pontosabb, lassabb.")},
            {"key": "poly_n",     "label": tr("Poly N"),           "type": "int",
             "default": 7, "range": (3, 9),
             "desc": tr("Polinomexpanzió szomszédsága. 7 = simább eredmény (Farneback 2003).")},
            {"key": "poly_sigma", "label": tr("Poly sigma"),       "type": "float",
             "default": 1.5, "range": (0.5, 3.0), "decimals": 1, "step": 0.1,
             "desc": tr("poly_n=7 esetén 1.5 ajánlott (Farneback 2003).")},
        ],
    },
    # ── Farneback – Részletes ─────────────────────────────────────────────────
    {
        "section": tr("💨  Optikai folyam – Részletes preset"),
        "prefix": "flow.presets.detailed",
        "keys": [
            {"key": "pyr_scale",  "label": tr("Piramis arány"),    "type": "float",
             "default": 0.5, "range": (0.1, 0.9), "decimals": 1, "step": 0.1,
             "desc": tr("Szintenkénti méretcsökkentés aránya. 0.5 = klasszikus piramis.")},
            {"key": "levels",     "label": tr("Piramisszintek"),   "type": "int",
             "default": 5, "range": (1, 8),
             "desc": tr("Több szint → nagyobb elmozdulás is nyomozható. Részletes: 5.")},
            {"key": "winsize",    "label": tr("Ablakméret (px)"),  "type": "int",
             "default": 33, "range": (5, 63),
             "desc": tr("Nagyobb ablak → simább, de kevésbé éles flow. Részletes: 33.")},
            {"key": "iterations", "label": tr("Iterációk"),        "type": "int",
             "default": 5, "range": (1, 10),
             "desc": tr("Iterációk száma szintenként. Részletes: 5 → legjobb minőség.")},
            {"key": "poly_n",     "label": tr("Poly N"),           "type": "int",
             "default": 7, "range": (3, 9),
             "desc": tr("Polinomexpanzió szomszédsága. 7 = simább eredmény (Farneback 2003).")},
            {"key": "poly_sigma", "label": tr("Poly sigma"),       "type": "float",
             "default": 1.5, "range": (0.5, 3.0), "decimals": 1, "step": 0.1,
             "desc": tr("poly_n=7 esetén 1.5 ajánlott (Farneback 2003).")},
        ],
    },
]


# ══════════════════════════════════════════════════════════════════════════════
#  ColorSwatch – szín megjelenítő + QColorDialog
# ══════════════════════════════════════════════════════════════════════════════

class ColorSwatch(QPushButton):
    """Kattintásra QColorDialog, jobb klikk: másolás / visszaállítás."""

    value_changed = pyqtSignal(str)   # #RRGGBB

    def __init__(self, hex_color: str, default: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._default = default
        self._color   = QColor(hex_color) if QColor(hex_color).isValid() else QColor(default)
        self.setFixedSize(70, 24)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(tr("Kattints a szín kiválasztásához\nJobb klikk: másolás / visszaállítás"))
        self._refresh_style()
        self.clicked.connect(self._pick)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._context_menu)

    # ── visuals ───────────────────────────────────────────────────────────────

    def _refresh_style(self):
        bg   = self._color.name()
        lum  = 0.299 * self._color.red() + 0.587 * self._color.green() + 0.114 * self._color.blue()
        fg   = "#000" if lum > 140 else "#fff"
        self.setStyleSheet(
            f"QPushButton{{background:{bg}; color:{fg}; border:1px solid #666;"
            f"border-radius:3px; font-size:9px; padding:0px;}}"
            f"QPushButton:hover{{border:2px solid #bbb;}}"
        )
        self.setText(bg)

    # ── slots ─────────────────────────────────────────────────────────────────

    def _pick(self):
        c = QColorDialog.getColor(self._color, self, "Szín kiválasztása")
        if c.isValid():
            self._color = c
            self._refresh_style()
            self.value_changed.emit(c.name())

    def _context_menu(self, pos):
        menu = QMenu(self)
        menu.addAction(
            f"📋  Hex kód másolása  ( {self._color.name()} )",
            lambda: QApplication.clipboard().setText(self._color.name()))
        menu.addSeparator()
        menu.addAction(
            f"↺  Visszaállítás alapértelmezettre  ( {self._default} )",
            self._reset)
        menu.exec(self.mapToGlobal(pos))

    def _reset(self):
        self._color = QColor(self._default)
        self._refresh_style()
        self.value_changed.emit(self._default)

    # ── public API ────────────────────────────────────────────────────────────

    def get_value(self) -> str:
        return self._color.name()

    def set_value(self, hex_color: str):
        c = QColor(hex_color)
        if c.isValid():
            self._color = c
            self._refresh_style()


# ══════════════════════════════════════════════════════════════════════════════
#  RGBAWidget – 4 csúszka + előnézet
# ══════════════════════════════════════════════════════════════════════════════

class RGBAWidget(QWidget):
    """R / G / B / A csúszkák + négyzet előnézet."""

    value_changed = pyqtSignal(list)   # [R, G, B, A]

    _CH_COLORS = ("#e55", "#5c5", "#55e", "#aaa")

    def __init__(self, rgba: Tuple, default: Tuple,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._default = tuple(int(x) for x in default)
        self._rgba    = list(int(x) for x in rgba)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        # Preview swatch
        self._preview = QLabel()
        self._preview.setFixedSize(28, 24)
        self._preview.setToolTip(tr("Jobb klikk: visszaállítás alapértelmezettre"))
        self._preview.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._preview.customContextMenuRequested.connect(self._context_menu)
        lay.addWidget(self._preview)

        # 4 sliders
        self._sliders: List[QSlider] = []
        for i, (ch, col) in enumerate(zip("RGBA", self._CH_COLORS)):
            col_lay = QVBoxLayout()
            col_lay.setSpacing(0)
            col_lay.setContentsMargins(0, 0, 0, 0)

            ch_lbl = QLabel(ch)
            ch_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            ch_lbl.setStyleSheet(f"color:{col}; font-size:8px; font-weight:bold;")
            col_lay.addWidget(ch_lbl)

            sl = QSlider(Qt.Orientation.Horizontal)
            sl.setRange(0, 255)
            sl.setValue(self._rgba[i])
            sl.setFixedWidth(52)
            sl.setFixedHeight(16)
            sl.setStyleSheet(
                f"QSlider::handle:horizontal{{background:{col}; border-radius:4px;"
                f"width:10px; height:10px;}}"
            )
            sl.valueChanged.connect(lambda v, idx=i: self._slider_moved(idx, v))
            col_lay.addWidget(sl)

            self._sliders.append(sl)
            lay.addLayout(col_lay)

        # Value label
        self._val_lbl = QLabel()
        self._val_lbl.setStyleSheet("color:#666; font-size:9px;")
        self._val_lbl.setFixedWidth(90)
        lay.addWidget(self._val_lbl)
        lay.addStretch()

        self._refresh_preview()

    # ── internals ─────────────────────────────────────────────────────────────

    def _slider_moved(self, idx: int, value: int):
        self._rgba[idx] = value
        self._refresh_preview()
        self.value_changed.emit(list(self._rgba))

    def _refresh_preview(self):
        r, g, b, a = self._rgba
        self._preview.setStyleSheet(
            f"background: rgba({r},{g},{b},{a});"
            "border: 1px solid #555; border-radius: 3px;"
        )
        self._val_lbl.setText(f"[{r},{g},{b},{a}]")

    def _context_menu(self, pos):
        menu = QMenu(self)
        menu.addAction("↺  Visszaállítás alapértelmezettre", self._reset)
        menu.exec(self._preview.mapToGlobal(pos))

    def _reset(self):
        self._rgba = list(self._default)
        for i, sl in enumerate(self._sliders):
            sl.blockSignals(True)
            sl.setValue(self._rgba[i])
            sl.blockSignals(False)
        self._refresh_preview()
        self.value_changed.emit(list(self._rgba))

    # ── public API ────────────────────────────────────────────────────────────

    def get_value(self) -> List[int]:
        return list(self._rgba)

    def set_value(self, rgba: Tuple):
        self._rgba = [int(x) for x in rgba]
        for i, sl in enumerate(self._sliders):
            sl.blockSignals(True)
            sl.setValue(self._rgba[i])
            sl.blockSignals(False)
        self._refresh_preview()


# ══════════════════════════════════════════════════════════════════════════════
#  TOML helpers
# ══════════════════════════════════════════════════════════════════════════════

def _toml_val(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        s = f"{v:.6f}".rstrip("0")
        return s if not s.endswith(".") else s + "0"
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(str(int(x)) for x in v) + "]"
    return str(v)


def _load_toml_file(path: pathlib.Path) -> dict:
    if sys.version_info >= (3, 11):
        import tomllib
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    else:
        import tomli                                     # type: ignore[import]
        with open(path, "rb") as fh:
            return tomli.load(fh)


def _flatten(node: dict, prefix: str = "") -> Dict[str, Any]:
    """Nested dict → flat {dot.key: value}."""
    out: Dict[str, Any] = {}
    for k, v in node.items():
        full = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, full))
        else:
            out[full] = v
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  ConfigEditorTab – a főwidget
# ══════════════════════════════════════════════════════════════════════════════

class ConfigEditorTab(QWidget):
    """Vizuális konfigurációs szerkesztő – ArchMorph Professional."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._widgets: Dict[str, QWidget] = {}   # full_key → widget
        self._build_ui()
        self._load_from_config()

    # ── UI építés ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 6)
        root.setSpacing(5)

        # ── Profil sáv ────────────────────────────────────────────────────────
        prof_frame = QFrame()
        prof_frame.setStyleSheet("QFrame{background:#1e232b; border-radius:4px; padding:2px;}")
        prof_lay = QHBoxLayout(prof_frame)
        prof_lay.setContentsMargins(8, 4, 8, 4)
        prof_lay.setSpacing(6)

        prof_lay.addWidget(QLabel(tr("💾  Profil:")))

        self._prof_combo = QComboBox()
        self._prof_combo.setMinimumWidth(180)
        self._prof_combo.setEditable(True)
        self._prof_combo.setPlaceholderText("profil neve…")
        self._prof_combo.lineEdit().returnPressed.connect(self._save_profile)
        prof_lay.addWidget(self._prof_combo)

        for icon, tip, slot in [
            ("💾 Mentés",  "Aktuális beállítások mentése profilként", self._save_profile),
            ("📂 Betöltés","Kiválasztott profil betöltése",           self._load_profile),
            ("🗑 Törlés",  "Kiválasztott profil törlése",             self._delete_profile),
        ]:
            b = QPushButton(icon)
            b.setToolTip(tip)
            b.setFixedHeight(26)
            b.clicked.connect(slot)
            prof_lay.addWidget(b)

        prof_lay.addSpacing(20)

        btn_reset = QPushButton(tr("↺  Minden visszaállítása"))
        btn_reset.setToolTip(tr("Minden értéket visszaállít az alapértelmezetthez"))
        btn_reset.setStyleSheet(
            "QPushButton{color:#e8a020;} QPushButton:hover{color:#ffc060;}")
        btn_reset.clicked.connect(self._reset_all)
        prof_lay.addWidget(btn_reset)

        prof_lay.addStretch()

        self._cfg_path_lbl = QLabel(f"📄  {_CONFIG_FILE.name}")
        self._cfg_path_lbl.setStyleSheet("color:#555; font-size:10px;")
        self._cfg_path_lbl.setToolTip(str(_CONFIG_FILE))
        prof_lay.addWidget(self._cfg_path_lbl)

        root.addWidget(prof_frame)

        # ── Scroll area ───────────────────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        self._content_lay = QVBoxLayout(content)
        self._content_lay.setContentsMargins(2, 4, 6, 4)
        self._content_lay.setSpacing(6)

        for sec in _SCHEMA:
            self._add_section(sec)

        self._content_lay.addStretch()
        scroll.setWidget(content)
        root.addWidget(scroll, 1)

        # ── Apply sáv ─────────────────────────────────────────────────────────
        apply_frame = QFrame()
        apply_frame.setStyleSheet("QFrame{background:#1e232b; border-radius:4px;}")
        apply_lay = QHBoxLayout(apply_frame)
        apply_lay.setContentsMargins(10, 4, 10, 4)

        self._status_lbl = QLabel()
        self._status_lbl.setStyleSheet("font-size:11px;")
        apply_lay.addWidget(self._status_lbl)

        apply_lay.addStretch()

        note = QLabel(tr("ℹ  Színek újraindítás után lépnek életbe."))
        note.setStyleSheet("color:#5a8a5a; font-size:10px;")
        apply_lay.addWidget(note)

        apply_lay.addSpacing(10)

        self._btn_save = QPushButton(tr("💾  Mentés   Ctrl+S"))
        self._btn_save.setStyleSheet(
            "QPushButton{background:#2e7d32; color:#eee; font-weight:bold;"
            "padding:4px 14px; border-radius:4px; border:none;}"
            "QPushButton:hover{background:#388e3c;}")
        self._btn_save.setShortcut(QKeySequence("Ctrl+S"))
        self._btn_save.clicked.connect(self._save_config)
        apply_lay.addWidget(self._btn_save)

        root.addWidget(apply_frame)

        self._refresh_profile_list()

    # ── Szekció hozzáadása ────────────────────────────────────────────────────

    def _add_section(self, sec: dict):
        group = QGroupBox(sec["section"])
        group.setStyleSheet(
            "QGroupBox{font-weight:bold; border:1px solid #2e3540;"
            "border-radius:4px; margin-top:8px; padding-top:2px;}"
            "QGroupBox::title{subcontrol-origin:margin; left:10px; color:#7aadcc;}"
        )
        form_lay = QVBoxLayout(group)
        form_lay.setContentsMargins(8, 6, 8, 6)
        form_lay.setSpacing(3)

        prefix = sec["prefix"]
        for kd in sec["keys"]:
            self._add_key_row(form_lay, prefix, kd)

        self._content_lay.addWidget(group)

    def _add_key_row(self, parent_lay: QVBoxLayout, prefix: str, kd: dict):
        full_key = f"{prefix}.{kd['key']}"
        current  = cfg(full_key, kd["default"])
        widget   = self._make_widget(kd, current)
        self._widgets[full_key] = widget

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        desc = kd.get("desc", "")

        # Label
        lbl = QLabel(kd["label"] + ":")
        lbl.setFixedWidth(188)
        lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lbl.setStyleSheet("color:#aaa; font-size:12px;")
        if desc:
            lbl.setToolTip(desc)
            lbl.setCursor(Qt.CursorShape.WhatsThisCursor)
        row.addWidget(lbl)

        # Widget – tooltip a bemenethez is
        if desc:
            widget.setToolTip(desc)
        row.addWidget(widget)

        # Description (halvány szöveg a jobb oldalon)
        if desc:
            desc_lbl = QLabel(desc)
            desc_lbl.setStyleSheet("color:#4a6070; font-size:10px;")
            desc_lbl.setWordWrap(False)
            desc_lbl.setSizePolicy(QSizePolicy.Policy.Expanding,
                                   QSizePolicy.Policy.Preferred)
            row.addWidget(desc_lbl, 1)
        else:
            row.addStretch(1)

        parent_lay.addLayout(row)

    # ── Widget gyártó ─────────────────────────────────────────────────────────

    def _make_widget(self, kd: dict, current: Any) -> QWidget:
        ktype   = kd["type"]
        default = kd["default"]

        if ktype == "color":
            w = ColorSwatch(str(current), str(default))
            w.value_changed.connect(self._mark_modified)
            return w

        if ktype == "rgba":
            rgba = tuple(current) if isinstance(current, (list, tuple)) else default
            w    = RGBAWidget(rgba, tuple(default))
            w.value_changed.connect(self._mark_modified)
            return w

        if ktype == "bool":
            w = QCheckBox()
            w.setChecked(bool(current))
            w.toggled.connect(self._mark_modified)
            return w

        if ktype == "choice":
            w = QComboBox()
            choices = kd.get("choices", [])
            w.addItems([str(c) for c in choices])
            if str(current) in [str(c) for c in choices]:
                w.setCurrentText(str(current))
            w.currentIndexChanged.connect(self._mark_modified)
            return w

        if ktype == "int":
            w = QSpinBox()
            lo, hi = kd.get("range", (0, 9999))
            w.setRange(lo, hi)
            w.setValue(int(current))
            w.setFixedWidth(90)
            # Right-click: reset
            w.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            w.customContextMenuRequested.connect(
                lambda pos, ww=w, d=default: self._num_menu(pos, ww, d))
            w.valueChanged.connect(self._mark_modified)
            return w

        if ktype == "float":
            w = QDoubleSpinBox()
            lo, hi = kd.get("range", (0.0, 1.0))
            w.setRange(lo, hi)
            w.setDecimals(kd.get("decimals", 2))
            w.setSingleStep(kd.get("step", 0.01))
            w.setValue(float(current))
            w.setFixedWidth(110)
            w.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            w.customContextMenuRequested.connect(
                lambda pos, ww=w, d=default: self._num_menu(pos, ww, d))
            w.valueChanged.connect(self._mark_modified)
            return w

        # Fallback
        lbl = QLabel(str(current))
        lbl.setStyleSheet("color:#888;")
        return lbl

    # ── Jobb klikk spinboxon ──────────────────────────────────────────────────

    def _num_menu(self, pos, widget: QWidget, default: Any):
        menu = QMenu(widget)
        menu.addAction(
            f"↺  Visszaállítás alapértelmezettre  ( {default} )",
            lambda: (
                (widget.setValue(int(default))
                 if isinstance(widget, QSpinBox)
                 else widget.setValue(float(default))),
                self._mark_modified(),
            )
        )
        menu.exec(widget.mapToGlobal(pos))

    # ── Módosítás jelző ───────────────────────────────────────────────────────

    def _mark_modified(self, *_):
        self._status_lbl.setText(tr("⚠  Nem mentett változtatások"))
        self._status_lbl.setStyleSheet("color:#e8a020; font-size:11px;")

    def _mark_saved(self, msg: str = "✓  Mentve"):
        self._status_lbl.setText(msg)
        self._status_lbl.setStyleSheet("color:#5c5; font-size:11px;")

    # ── Értékek kiolvasása / beállítása ───────────────────────────────────────

    def _collect(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, w in self._widgets.items():
            if isinstance(w, ColorSwatch):
                out[key] = w.get_value()
            elif isinstance(w, RGBAWidget):
                out[key] = w.get_value()
            elif isinstance(w, QCheckBox):
                out[key] = w.isChecked()
            elif isinstance(w, QComboBox):
                out[key] = w.currentText()
            elif isinstance(w, QSpinBox):
                out[key] = w.value()
            elif isinstance(w, QDoubleSpinBox):
                out[key] = w.value()
        return out

    def _apply_flat(self, flat: Dict[str, Any]):
        for key, value in flat.items():
            w = self._widgets.get(key)
            if w is None:
                continue
            try:
                if isinstance(w, ColorSwatch):
                    w.set_value(str(value))
                elif isinstance(w, RGBAWidget):
                    if isinstance(value, (list, tuple)) and len(value) == 4:
                        w.set_value(tuple(value))
                elif isinstance(w, QCheckBox):
                    w.setChecked(bool(value))
                elif isinstance(w, QComboBox):
                    w.setCurrentText(str(value))
                elif isinstance(w, QSpinBox):
                    w.setValue(int(value))
                elif isinstance(w, QDoubleSpinBox):
                    w.setValue(float(value))
            except Exception as e:
                import traceback; traceback.print_exc()

    def _load_from_config(self):
        """Betölti a jelenlegi cfg() értékeket minden widgetbe."""
        for sec in _SCHEMA:
            prefix = sec["prefix"]
            for kd in sec["keys"]:
                full_key = f"{prefix}.{kd['key']}"
                current  = cfg(full_key, kd["default"])
                self._apply_flat({full_key: current})
        self._status_lbl.setText(tr(""))

    # ── TOML írás ─────────────────────────────────────────────────────────────

    def _values_to_toml(self, flat: Dict[str, Any]) -> str:
        """
        Flat dict → TOML szöveg.
        A schema sorrendjében írja a szekciókat, inline kommentekkel.
        """
        lines: List[str] = [
            "# ArchMorph Professional – konfigurációs fájl",
            "# Generálva a vizuális szerkesztővel.",
            "# Teljes dokumentáció: archmorph_config.toml (eredeti sablon)",
            "",
        ]
        for sec in _SCHEMA:
            prefix = sec["prefix"]
            lines.append(f"\n[{prefix}]")
            lines.append(f"# {sec['section'].lstrip('🎨🟠🖼⬜🔷🎬📐🔬💿🔭🔍🎯📤💨 ')}")
            for kd in sec["keys"]:
                key      = kd["key"]
                full_key = f"{prefix}.{key}"
                value    = flat.get(full_key, kd["default"])
                desc     = kd.get("desc", "")
                label    = kd["label"]
                comment  = desc.split("\n")[0] if desc else label
                lines.append(f"# {comment}")
                lines.append(f"{key} = {_toml_val(value)}")
        return "\n".join(lines) + "\n"

    # ── Mentés / betöltés ─────────────────────────────────────────────────────

    def _save_config(self):
        """Menti az archmorph_config.toml fájlt."""
        flat = self._collect()
        toml_text = self._values_to_toml(flat)
        try:
            _CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
            _CONFIG_FILE.write_text(toml_text, encoding="utf-8")
            cfg_reload()
            self._mark_saved(f"✓  Mentve → {_CONFIG_FILE.name}")
        except Exception as exc:
            QMessageBox.critical(self, "Mentési hiba", str(exc))

    # ── Profil kezelés ────────────────────────────────────────────────────────

    def _refresh_profile_list(self):
        current_text = self._prof_combo.currentText()
        self._prof_combo.clear()
        if _PROFILES_DIR.exists():
            for f in sorted(_PROFILES_DIR.glob("*.toml")):
                self._prof_combo.addItem(f.stem)
        if current_text:
            self._prof_combo.setCurrentText(current_text)

    def _save_profile(self):
        name = self._prof_combo.currentText().strip()
        if not name:
            QMessageBox.warning(self, "Profil", "Add meg a profil nevét!")
            return
        # Sanitize name
        safe = "".join(c for c in name if c.isalnum() or c in "- _").strip()
        if not safe:
            QMessageBox.warning(self, "Profil", "Érvénytelen profil név.")
            return
        _PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        path = _PROFILES_DIR / f"{safe}.toml"
        flat = self._collect()
        toml_text = self._values_to_toml(flat)
        try:
            path.write_text(toml_text, encoding="utf-8")
            self._refresh_profile_list()
            self._prof_combo.setCurrentText(safe)
            self._mark_saved(f"✓  Profil mentve: {safe}.toml")
        except Exception as exc:
            QMessageBox.critical(self, "Mentési hiba", str(exc))

    def _load_profile(self):
        name = self._prof_combo.currentText().strip()
        path = _PROFILES_DIR / f"{name}.toml"
        if not path.exists():
            # Try exact filename
            path2 = _PROFILES_DIR / name
            if path2.exists():
                path = path2
            else:
                QMessageBox.warning(self, "Profil", f"Nem található: {name}.toml")
                return
        try:
            raw  = _load_toml_file(path)
            flat = _flatten(raw)
            self._apply_flat(flat)
            self._mark_modified()
            self._status_lbl.setText(f"📂  Profil betöltve: {name}  – mentsd el az alkalmazáshoz")
            self._status_lbl.setStyleSheet("color:#7bc; font-size:11px;")
        except Exception as exc:
            QMessageBox.critical(self, "Betöltési hiba", str(exc))

    def _delete_profile(self):
        name = self._prof_combo.currentText().strip()
        path = _PROFILES_DIR / f"{name}.toml"
        if not path.exists():
            return
        ans = QMessageBox.question(
            self, "Profil törlése",
            f'Biztosan törlöd a \u201e{name}\u201c profilt?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if ans == QMessageBox.StandardButton.Yes:
            path.unlink()
            self._refresh_profile_list()

    def _reset_all(self):
        ans = QMessageBox.question(
            self, "Visszaállítás",
            "Minden beállítást visszaállít az alapértelmezett értékre.\n\nBiztosan?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if ans != QMessageBox.StandardButton.Yes:
            return
        default_flat = {
            f"{sec['prefix']}.{kd['key']}": kd["default"]
            for sec in _SCHEMA
            for kd in sec["keys"]
        }
        self._apply_flat(default_flat)
        self._mark_modified()
