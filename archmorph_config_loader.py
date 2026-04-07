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


def save_config(key_path: str, value: Any) -> None:
    """
    Egy kulcsot ír/frissít a TOML fájlban.

    Csak egyszerű dot-notációjú kulcsokat kezel (pl. "ui.language").
    Ha a fájl nem létezik vagy nem írható, csendesen kihagyja.
    """
    _ensure_loaded()

    # Frissítjük a memóriabeli dict-et
    parts = key_path.split(".")
    node: Any = _data
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value

    # Visszaírás TOML-ba (csak ha a fájl már létezik)
    if not _CONFIG_FILE.exists():
        return
    try:
        # Egyszerű sor-alapú frissítés: ha a sor tartalmazza a kulcsot,
        # lecseréljük; különben a fájl végéhez fűzzük.
        key_leaf = parts[-1]
        section  = ".".join(parts[:-1])

        lines = _CONFIG_FILE.read_text(encoding="utf-8").splitlines(keepends=True)
        in_section = (section == "")
        target_header = f"[{section}]" if section else None
        replaced = False

        for i, line in enumerate(lines):
            stripped = line.strip()
            if target_header and stripped == target_header:
                in_section = True
                continue
            if in_section and stripped.startswith("[") and stripped != target_header:
                in_section = False
            if in_section and stripped.startswith(f"{key_leaf}"):
                val_str = _toml_value(value)
                lines[i] = f"{key_leaf} = {val_str}\n"
                replaced = True
                break

        if not replaced:
            val_str = _toml_value(value)
            if target_header:
                lines.append(f"\n[{section}]\n{key_leaf} = {val_str}\n")
            else:
                lines.append(f"{key_leaf} = {val_str}\n")

        _CONFIG_FILE.write_text("".join(lines), encoding="utf-8")
    except Exception:
        pass  # Írási hiba: csendesen kihagyja


def _toml_value(v: Any) -> str:
    """Egyszerű Python → TOML érték konverzió."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        escaped = v.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return str(v)
