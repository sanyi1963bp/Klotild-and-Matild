"""
archmorph_config_loader.py
══════════════════════════
ArchMorph Professional – konfigurációs fájl betöltő

Betöltési sorrend:
  1. Ugyanabban a könyvtárban keresi az "archmorph_config.toml" fájlt.
  2. Ha nem létezik → a program beépített alapértelmezett értékeket használ.
  3. Ha hibás TOML szintaxis van → hibaüzenet a konzolra, beépített értékek.

TOML könyvtár:
  Python 3.11+  → beépített tomllib (semmi teendő)
  Python 3.7–3.10 → pip install tomli

Használat más modulokban:
  from archmorph_config_loader import cfg, cfg_rgba

  szin = cfg("ui.colors.points.normal", "#ff8a3d")   # str visszatérő
  rgba = cfg_rgba("ui.colors.roi.fill_rgba", (255, 153, 0, 22))  # tuple
"""
from __future__ import annotations

import sys
import pathlib
from typing import Any, Tuple

# ── TOML könyvtár kiválasztása ────────────────────────────────────────────────

_TOML_AVAILABLE = False

if sys.version_info >= (3, 11):
    import tomllib as _toml_lib   # beépített Python 3.11+
    _TOML_AVAILABLE = True
else:
    try:
        import tomli as _toml_lib  # type: ignore[no-redef]  # pip install tomli
        _TOML_AVAILABLE = True
    except ImportError:
        _toml_lib = None  # type: ignore[assignment]

# ── Konfigurációs fájl elérési útja ──────────────────────────────────────────

_CONFIG_FILE: pathlib.Path = pathlib.Path(__file__).parent / "archmorph_config.toml"

# ── Belső állapot ─────────────────────────────────────────────────────────────

_data: dict = {}
_loaded: bool = False


def _ensure_loaded() -> None:
    """Fájlt pontosan egyszer tölti be (lazy loading)."""
    global _data, _loaded
    if _loaded:
        return
    _loaded = True

    if not _TOML_AVAILABLE:
        print(
            "[ArchMorph Config] ⚠  TOML könyvtár nem elérhető.\n"
            "  Python 3.11+: tomllib beépített – nem kell telepíteni.\n"
            "  Python 3.7–3.10: futtasd:  pip install tomli\n"
            "  → Beépített alapértelmezett értékek lesznek használva."
        )
        return

    if not _CONFIG_FILE.exists():
        print(
            f"[ArchMorph Config] ℹ  Konfigurációs fájl nem található:\n"
            f"  {_CONFIG_FILE}\n"
            "  → Beépített alapértelmezett értékek lesznek használva.\n"
            "  (Ez normális az első indításnál.)"
        )
        return

    try:
        with open(_CONFIG_FILE, "rb") as fh:
            _data = _toml_lib.load(fh)
        print(f"[ArchMorph Config] ✓  Betöltve: {_CONFIG_FILE}")
    except Exception as exc:
        print(
            f"[ArchMorph Config] ✗  Hiba a betöltésnél: {exc}\n"
            "  Ellenőrizd a TOML szintaxist (pl. VS Code-dal).\n"
            "  → Beépített alapértelmezett értékek lesznek használva."
        )
        _data = {}


# ── Nyilvános API ─────────────────────────────────────────────────────────────

def cfg(key_path: str, default: Any) -> Any:
    """
    Dot-notációval olvas a konfigból.

    Példa:
        cfg("ui.colors.points.normal", "#ff8a3d")
        cfg("match.sift.ratio_threshold", 0.80)
        cfg("ui.sizes.point_radius_normal", 6)

    Ha a kulcs nem létezik, vagy a fájl nem tölthető be,
    a ``default`` értékkel tér vissza (nincs kivétel).
    """
    _ensure_loaded()
    parts = key_path.split(".")
    node: Any = _data
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def cfg_rgba(key_path: str, default: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """
    RGBA lista olvasása: [R, G, B, A]  →  (R, G, B, A) tuple.

    Példa:
        r, g, b, a = cfg_rgba("ui.colors.roi.fill_rgba", (255, 153, 0, 22))
        color = QColor(*cfg_rgba("ui.colors.points.fill.normal_rgba", (255,138,61,70)))
    """
    val = cfg(key_path, default)
    if isinstance(val, (list, tuple)) and len(val) == 4:
        try:
            return (int(val[0]), int(val[1]), int(val[2]), int(val[3]))
        except (TypeError, ValueError):
            pass
    return default


def reload() -> None:
    """Kényszeríti a konfig újratöltését (pl. futás közben szerkesztés után)."""
    global _loaded
    _loaded = False
    _data.clear()
    _ensure_loaded()
