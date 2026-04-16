"""
archmorph.py  –  ArchMorph Professional  v0.5.0
================================================
Főfájl. A pontszerkesztő logika külön modulban van:  point_editor.py

Modulstruktúra
--------------
    archmorph.py     – fő alkalmazás (ez a fájl)
    point_editor.py  – PointEditorCanvas + PointEditorWidget

Indítás
-------
    python archmorph.py
"""
from __future__ import annotations

import sys
import math
import json
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np

# Opcionális függőségek
try:
    import cv2
except Exception:
    cv2 = None

try:
    import torch
except Exception:
    torch = None

try:
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import rbd
    _HAS_LIGHTGLUE = True
except Exception:
    LightGlue = SuperPoint = rbd = None
    _HAS_LIGHTGLUE = False

try:
    from lightglue import DISK as _DISK_extractor
    _HAS_DISK = True
except Exception:
    _DISK_extractor = None
    _HAS_DISK = False

try:
    from kornia.feature import LoFTR as _KorniaLoFTR
    _HAS_LOFTR = True
except Exception:
    _KorniaLoFTR = None
    _HAS_LOFTR = False

try:
    from PIL import Image as _PilImage
    _HAS_PIL = True
except Exception:
    _PilImage = None
    _HAS_PIL = False

from PyQt6.QtCore import Qt, QSize, QRectF, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QAction, QColor, QFont, QImage, QPainter, QPen, QPixmap,
)
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDockWidget, QDoubleSpinBox,
    QFileDialog, QFormLayout, QFrame, QGroupBox, QInputDialog, QLabel,
    QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QMenu, QMessageBox, QProgressDialog,
    QPushButton, QScrollArea, QSizePolicy, QSlider, QSplitter, QSpinBox,
    QStackedWidget, QStatusBar, QTabWidget, QTextEdit, QToolBar,
    QToolButton, QVBoxLayout, QHBoxLayout, QWidget,
)

# Saját modulok
from point_editor  import PointEditorWidget
from theme_editor  import ThemeEditor
try:
    from archmorph_config_loader import cfg
except ImportError:
    def cfg(key, default):  return default   # type: ignore[misc]

try:
    from config_editor import ConfigEditorTab
    _HAS_CONFIG_EDITOR = True
except ImportError:
    ConfigEditorTab = None  # type: ignore[assignment,misc]
    _HAS_CONFIG_EDITOR = False


try:
    from gcp_dialog import GCPDialog
    _HAS_GCP_DIALOG = True
except ImportError:
    GCPDialog = None         # type: ignore[assignment,misc]
    _HAS_GCP_DIALOG = False


try:
    from TRANSLATIONS import tr, set_language, get_language
except ImportError:
    def tr(text: str) -> str: return text
    def set_language(lang: str) -> None: pass
    def get_language() -> str: return "hu"

APP_NAME    = "ArchMorph Professional"
APP_VERSION = "0.5.0"


# ════════════════════════════════════════════════════════════════════════════
#  Adatmodellek
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class AppSettings:
    help_visible:            bool  = True
    dark_mode:               bool  = True
    matcher_backend:         str   = "SuperPoint + LightGlue"
    warp_intensity:          float = 0.60
    blend_duration:          float = 2.0
    triangulation_density:   int   = 48
    export_fps:              int   = 30
    export_quality:          str   = "High"
    use_depth:               bool  = False
    use_inpaint:             bool  = False
    max_keypoints:           int   = 2048
    match_threshold:         float = 0.10
    use_ransac_filter:       bool  = True
    ransac_reproj_threshold: float = 4.0
    display_match_limit:     int   = 150
    overlay_alpha:           float = 0.50
    alignment_view_mode:     str   = tr("Overlay")
    force_cpu_matching:      bool  = False
    auto_fallback_to_cpu:    bool  = True


@dataclass
class ProjectState:
    # ── Meta ────────────────────────────────────────────────────────────────
    project_name:           str                        = ""   # ember-olvasható projektnév
    notes:                  str                        = ""   # szabad szöveges megjegyzés
    created_at:             str                        = ""   # ISO 8601 timestamp
    modified_at:            str                        = ""   # ISO 8601 timestamp

    # ── Képek ───────────────────────────────────────────────────────────────
    image_a_path:           Optional[Path]             = None
    image_b_path:           Optional[Path]             = None
    image_a:                Optional[np.ndarray]       = None
    image_b:                Optional[np.ndarray]       = None

    # ── Manuális morfpontok ─────────────────────────────────────────────────
    anchor_points_a:        List[List[float]]          = field(default_factory=list)
    anchor_points_b:        List[List[float]]          = field(default_factory=list)
    # Vonalak (pl. épületélek) – rendereléskor kerülnek felosztásra pontpárokká.
    # Formátum: [{"pts_a": [[x,y],...], "pts_b": [[x,y],...]}, ...]
    polylines:              List[dict]                 = field(default_factory=list)

    # ── Automata illesztés ─────────────────────────────────────────────────
    raw_matches_a:          List[List[float]]          = field(default_factory=list)
    raw_matches_b:          List[List[float]]          = field(default_factory=list)
    raw_inlier_mask:        List[bool]                 = field(default_factory=list)
    homography_matrix:      Optional[List[List[float]]] = None

    # ── Levezetett/cache adatok (nem mentett) ───────────────────────────────
    aligned_image_a_to_b:   Optional[np.ndarray]       = None
    aligned_overlay_preview: Optional[np.ndarray]      = None
    exclusion_masks:        Dict[str, Any]             = field(default_factory=dict)

    # ── Képmódosítás jelzők ─────────────────────────────────────────────────
    # True ha a memóriában lévő kép eltér a image_*_path-on lévő fájltól
    # (pl. GCP warp vagy crop után) → mentéskor külön fájlba kell írni
    image_a_is_modified:    bool                       = False
    image_b_is_modified:    bool                       = False
    # Eredeti képek GCP/dőlés-korrekció előtti állapota (visszavonáshoz)
    image_a_original:       Optional[np.ndarray]       = None
    image_b_original:       Optional[np.ndarray]       = None

    # ── Munkafolyamat-állapot ───────────────────────────────────────────────
    gcp_done:               bool                       = False
    auto_match_done:        bool                       = False


# ════════════════════════════════════════════════════════════════════════════
#  Segédfüggvények
# ════════════════════════════════════════════════════════════════════════════

def log_exception_text(prefix: str, exc: BaseException) -> str:
    return f"{prefix}\n{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"


def ensure_utf8_json_dump(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json_utf8(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def cv_imread_unicode_safe(path: Path) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("Az OpenCV (cv2) nincs telepítve.")
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is not None:
        return img
    raw = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"A kép nem olvasható: {path}")
    return img


def bgr_to_qpixmap(image_bgr: np.ndarray) -> QPixmap:
    if cv2 is None:
        rgb = image_bgr[:, :, ::-1].copy()
    else:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def get_torch_device(force_cpu: bool = False) -> str:
    if force_cpu or torch is None:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def bgr_to_torch_image(image_bgr: np.ndarray, device: str):
    if torch is None or cv2 is None:
        raise RuntimeError("PyTorch vagy OpenCV hiányzik.")
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return tensor.to(device)




def _validate_anchor_points(pts_a, pts_b) -> tuple:
    """Validate and return clean (pts_a, pts_b) or raise ValueError."""
    if not isinstance(pts_a, list) or not isinstance(pts_b, list):
        raise ValueError("Point lists must be lists.")
    if len(pts_a) != len(pts_b):
        raise ValueError(f"Point list length mismatch: {len(pts_a)} vs {len(pts_b)}")
    clean_a, clean_b = [], []
    for i, (pa, pb) in enumerate(zip(pts_a, pts_b)):
        if not (isinstance(pa, (list,tuple)) and len(pa)==2 and
                all(isinstance(v,(int,float)) for v in pa)):
            raise ValueError(f"Invalid point A[{i}]: {pa}")
        if not (isinstance(pb, (list,tuple)) and len(pb)==2 and
                all(isinstance(v,(int,float)) for v in pb)):
            raise ValueError(f"Invalid point B[{i}]: {pb}")
        clean_a.append((float(pa[0]), float(pa[1])))
        clean_b.append((float(pb[0]), float(pb[1])))
    return clean_a, clean_b

# ════════════════════════════════════════════════════════════════════════════
#  Illesztési algoritmusok
# ════════════════════════════════════════════════════════════════════════════

def run_superpoint_lightglue(
    image_a_bgr: np.ndarray,
    image_b_bgr: np.ndarray,
    *,
    max_keypoints:       int   = 2048,
    match_threshold:     float = 0.10,
    detection_threshold: float = 0.0005,
    nms_radius:          int   = 4,
    depth_confidence:    float = 0.95,
    width_confidence:    float = 0.99,
    force_cpu:           bool  = False,
    **_,                                   # extra paramétereket csendesen elnyel
) -> Tuple[np.ndarray, np.ndarray, str]:
    if torch is None:
        raise RuntimeError("A PyTorch nincs telepítve.")
    if LightGlue is None or SuperPoint is None or rbd is None:
        raise RuntimeError("A lightglue csomag nincs telepítve.")
    device    = get_torch_device(force_cpu=force_cpu)
    extractor = SuperPoint(
        max_num_keypoints   = max_keypoints,
        detection_threshold = detection_threshold,
        nms_radius          = nms_radius,
    ).eval().to(device)
    matcher   = LightGlue(
        features         = "superpoint",
        filter_threshold = match_threshold,
        depth_confidence = depth_confidence,
        width_confidence = width_confidence,
    ).eval().to(device)
    img0 = bgr_to_torch_image(image_a_bgr, device)
    img1 = bgr_to_torch_image(image_b_bgr, device)
    with torch.inference_mode():
        feats0 = extractor.extract(img0)
        feats1 = extractor.extract(img1)
        m01    = matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, m01 = [rbd(x) for x in (feats0, feats1, m01)]
        matches = m01["matches"]
        if matches is None or len(matches) == 0:
            return (np.empty((0, 2), np.float32),
                    np.empty((0, 2), np.float32), device)
        pts0 = feats0["keypoints"][matches[..., 0]].cpu().numpy().astype(np.float32)
        pts1 = feats1["keypoints"][matches[..., 1]].cpu().numpy().astype(np.float32)
        return pts0, pts1, device


def run_sift_opencv(
    image_a_bgr: np.ndarray,
    image_b_bgr: np.ndarray,
    *,
    max_keypoints:      int   = 2048,
    n_octave_layers:    int   = 3,
    contrast_threshold: float = 0.04,
    edge_threshold:     float = 10.0,
    sigma:              float = 1.6,
    ratio_threshold:    float = 0.75,
    **_,                                   # extra paramétereket csendesen elnyel
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    SIFT-alapú illesztés OpenCV-vel – GPU és PyTorch nélkül fut.

    Pontdetekció:  cv2.SIFT_create(nfeatures, nOctaveLayers,
                                   contrastThreshold, edgeThreshold, sigma)
    Matcher:       BFMatcher L2  +  Lowe-arány szűrés (ratio_threshold)
    Visszatérés:   (pts_a, pts_b, "cpu")
    """
    if cv2 is None:
        raise RuntimeError("Az OpenCV (cv2) nincs telepítve.")
    sift  = cv2.SIFT_create(
        nfeatures        = max_keypoints,
        nOctaveLayers    = n_octave_layers,
        contrastThreshold= contrast_threshold,
        edgeThreshold    = edge_threshold,
        sigma            = sigma,
    )
    gray0 = cv2.cvtColor(image_a_bgr, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(image_b_bgr, cv2.COLOR_BGR2GRAY)
    kp0, des0 = sift.detectAndCompute(gray0, None)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    _empty = (np.empty((0, 2), np.float32), np.empty((0, 2), np.float32), "cpu")
    if des0 is None or des1 is None or len(kp0) < 4 or len(kp1) < 4:
        return _empty
    bf  = cv2.BFMatcher(cv2.NORM_L2)
    raw = bf.knnMatch(des0, des1, k=2)
    good = []
    for pair in raw:
        if len(pair) == 2 and pair[0].distance < ratio_threshold * pair[1].distance:
            good.append(pair[0])
    if not good:
        return _empty
    pts0 = np.array([kp0[m.queryIdx].pt for m in good], dtype=np.float32)
    pts1 = np.array([kp1[m.trainIdx].pt for m in good], dtype=np.float32)
    return pts0, pts1, "cpu"


def run_disk_lightglue(
    image_a_bgr: np.ndarray,
    image_b_bgr: np.ndarray,
    *,
    max_keypoints:   int   = 2048,
    match_threshold: float = 0.10,
    force_cpu:       bool  = False,
    **_,                                   # extra paramétereket csendesen elnyel
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    DISK detektor + LightGlue matcher.
    Neurális, de más jellemzőkre tanítva mint SuperPoint –
    épületek ismétlődő mintáin (ablaksorok, kövezet) általában jobb.
    Ugyanaz a lightglue csomag kell hozzá, nincs extra telepítés.
    """
    if torch is None:
        raise RuntimeError("A PyTorch nincs telepítve.")
    if not _HAS_LIGHTGLUE or not _HAS_DISK:
        raise RuntimeError(
            "A DISK extractor nincs elérhető (lightglue >= 0.1 szükséges).")
    device    = get_torch_device(force_cpu=force_cpu)
    extractor = _DISK_extractor(max_num_keypoints=max_keypoints).eval().to(device)
    matcher   = LightGlue(features="disk",
                          filter_threshold=match_threshold).eval().to(device)
    img0 = bgr_to_torch_image(image_a_bgr, device)
    img1 = bgr_to_torch_image(image_b_bgr, device)
    with torch.inference_mode():
        feats0 = extractor.extract(img0)
        feats1 = extractor.extract(img1)
        m01    = matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, m01 = [rbd(x) for x in (feats0, feats1, m01)]
        matches = m01["matches"]
        if matches is None or len(matches) == 0:
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32), device
        pts0 = feats0["keypoints"][matches[..., 0]].cpu().numpy().astype(np.float32)
        pts1 = feats1["keypoints"][matches[..., 1]].cpu().numpy().astype(np.float32)
        return pts0, pts1, device


def run_loftr(
    image_a_bgr: np.ndarray,
    image_b_bgr: np.ndarray,
    *,
    force_cpu:            bool  = False,
    pretrained:           str   = tr("outdoor"),
    confidence_threshold: float = 0.5,
    **_,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    LoFTR (Detector-Free Local Feature Matching) via kornia.
    Detektor nélkül sűrű egyezéseket ad – különösen jó textúraszegény
    vagy erősen deformált képpárokon (pl. historikus + modern fotó).

    Telepítés: pip install kornia
    pretrained: tr("outdoor") (épületek, utcák) vagy tr("indoor") (belső terek)
    """
    if torch is None:
        raise RuntimeError("A PyTorch nincs telepítve.")
    if not _HAS_LOFTR:
        raise RuntimeError(
            "A kornia nincs telepítve.\n"
            "Telepítés: pip install kornia")
    if cv2 is None:
        raise RuntimeError("Az OpenCV (cv2) nincs telepítve.")
    device  = get_torch_device(force_cpu=force_cpu)
    matcher = _KorniaLoFTR(pretrained=pretrained).eval().to(device)

    def _prepare(bgr: np.ndarray):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        t    = torch.from_numpy(gray)[None, None].to(device)  # (1,1,H,W)
        return t

    img0 = _prepare(image_a_bgr)
    img1 = _prepare(image_b_bgr)
    with torch.inference_mode():
        out = matcher({"image0": img0, "image1": img1})
    kp0   = out["keypoints0"][0].cpu().numpy().astype(np.float32)
    kp1   = out["keypoints1"][0].cpu().numpy().astype(np.float32)
    conf  = out["confidence"][0].cpu().numpy()
    # Konfidencia-küszöb: csak a legjobb egyezések
    keep  = conf > confidence_threshold
    return kp0[keep], kp1[keep], device


def filter_matches_with_ransac(
    points_a: np.ndarray,
    points_b: np.ndarray,
    *,
    reproj_threshold: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if cv2 is None:
        raise RuntimeError("Az OpenCV (cv2) nincs telepítve.")
    if len(points_a) < 4:
        mask = np.ones(len(points_a), dtype=bool)
        return points_a.copy(), points_b.copy(), mask, None
    H, mask = cv2.findHomography(points_a, points_b, cv2.RANSAC, reproj_threshold)
    if mask is None or H is None:
        mask = np.zeros(len(points_a), dtype=bool)
        return (np.empty((0, 2), np.float32),
                np.empty((0, 2), np.float32), mask, None)
    mask = mask.reshape(-1).astype(bool)
    return points_a[mask], points_b[mask], mask, H


def warp_image_to_reference(src: np.ndarray, ref: np.ndarray,
                            H: np.ndarray) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("Az OpenCV (cv2) nincs telepítve.")
    rh, rw = ref.shape[:2]
    return cv2.warpPerspective(src, H, (rw, rh),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(0, 0, 0))


def crop_images_to_overlap(
    warped_a: np.ndarray,
    image_b:  np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Mindkét képet a valódi közös átfedési területre vágja.

    A warpPerspective után a warped_a szélein fekete sávok keletkezhetnek
    (ahol az A kép nem ért el), ezeket és az image_b megfelelő részeit
    ez a függvény eltávolítja.

    Visszatér: (cropped_a, cropped_b, x_offset, y_offset)
      x_offset, y_offset: a kivágás bal-felső sarka az eredeti B-koordinátákban
      (szükséges a GCP-inlier pontok eltolásához).
    Ha nincs érdemi átfedés, az eredeti képeket adja vissza 0,0 offsettel.
    """
    # Nem-fekete maszk: legalább az egyik csatornán > 10 értékű pixel
    def _nonblack(img: np.ndarray) -> np.ndarray:
        return np.any(img > 10, axis=2) if img.ndim == 3 else img > 10

    overlap = _nonblack(warped_a) & _nonblack(image_b)

    rows = np.any(overlap, axis=1)
    cols = np.any(overlap, axis=0)
    if not rows.any() or not cols.any():
        return warped_a, image_b, 0, 0

    r0 = int(np.where(rows)[0][0])
    r1 = int(np.where(rows)[0][-1]) + 1
    c0 = int(np.where(cols)[0][0])
    c1 = int(np.where(cols)[0][-1]) + 1

    return warped_a[r0:r1, c0:c1], image_b[r0:r1, c0:c1], c0, r0


def detect_tilt_angle(image_bgr: np.ndarray) -> float:
    """
    Megpróbálja meghatározni a kép dőlési szögét Hough-egyenesek alapján.

    A függőlegeshez közeli éleket keresi (±30° toleranciával), majd
    mediánnal becsüli a szükséges korrekciós szöget.

    Visszatér: szög fokokban (pozitív → óramutató járásával megegyező forgás).
               Ha nem talál egyenest, 0.0-t ad vissza.
    """
    if cv2 is None:
        return 0.0
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=80, minLineLength=60, maxLineGap=10)
    if lines is None:
        return 0.0

    angles: list[float] = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        if abs(dy) < 1e-6:
            continue  # vízszintes → kizárjuk
        angle_from_vertical = float(np.degrees(np.arctan2(dx, dy)))
        # Csak ±30°-on belüli vonalakat fogadunk el
        if abs(angle_from_vertical) <= 30.0:
            angles.append(angle_from_vertical)

    if not angles:
        return 0.0
    return float(np.median(angles))


def rotate_image_by_angle(image_bgr: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Elforgatja a képet `angle_deg` fokkal (óramutató irányában pozitív).
    A canvas mérete megnő, hogy semelyik sarok se vágódjon le.
    A hátterszín fekete (0, 0, 0).
    """
    if cv2 is None:
        return image_bgr
    h, w = image_bgr.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), -angle_deg, 1.0)

    # Új (nagyobb) vászonméret kiszámítása
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)

    # Fordítási korrekció (kép középre kerül az új vásznon)
    M[0, 2] += (new_w - w) / 2.0
    M[1, 2] += (new_h - h) / 2.0

    return cv2.warpAffine(image_bgr, M, (new_w, new_h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0, 0, 0))


# ─────────────────────────────────────────────────────────────────────────────

def blend_same_size_images(a: np.ndarray, b: np.ndarray,
                            *, alpha: float = 0.5) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("Az OpenCV (cv2) nincs telepítve.")
    if a.shape != b.shape:
        a = cv2.resize(a, (b.shape[1], b.shape[0]))
    alpha = float(max(0.0, min(1.0, alpha)))
    return cv2.addWeighted(a, alpha, b, 1.0 - alpha, 0.0)


# ════════════════════════════════════════════════════════════════════════════
#  Export / morph segédfüggvények
# ════════════════════════════════════════════════════════════════════════════

def _easing_linear(t: float) -> float:
    return t

def _easing_ease_in(t: float) -> float:
    return t * t

def _easing_ease_out(t: float) -> float:
    return 1.0 - (1.0 - t) ** 2

def _easing_ease_in_out(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)  # smoothstep

EASING_FUNCS: Dict[str, Any] = {
    "Lineáris":    _easing_linear,
    "Lassú start": _easing_ease_in,
    "Lassú vége":  _easing_ease_out,
    "S-görbe":     _easing_ease_in_out,
}


def interpolate_homography(H: np.ndarray, t: float) -> np.ndarray:
    """Lineáris interpoláció az egységmátrix és H között, majd normalizálás."""
    I  = np.eye(3, dtype=np.float64)
    Ht = (1.0 - t) * I + t * H.astype(np.float64)
    Ht /= Ht[2, 2]
    return Ht


def generate_morph_frames(
    img_a:     np.ndarray,
    img_b:     np.ndarray,
    H:         np.ndarray,
    n_frames:  int  = 30,
    easing:    str  = "S-görbe",
    ping_pong: bool = False,
) -> List[np.ndarray]:
    """
    Köztes képkockák listája az A→B morph animációhoz.
    H: img_a → img_b homográfia mátrix (3×3).
    Visszatér: BGr numpy tömbök listája.
    """
    if cv2 is None:
        raise RuntimeError("Az OpenCV (cv2) nincs telepítve.")
    func  = EASING_FUNCS.get(easing, _easing_linear)
    h, w  = img_b.shape[:2]
    n     = max(n_frames, 2)
    frames: List[np.ndarray] = []
    for i in range(n):
        t_raw = i / (n - 1)
        t     = func(t_raw)
        Ht    = interpolate_homography(H, t)
        wa    = cv2.warpPerspective(img_a, Ht, (w, h),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(0, 0, 0))
        frame = blend_same_size_images(wa, img_b, alpha=t)
        frames.append(frame)
    if ping_pong:
        frames = frames + frames[-2:0:-1]
    return frames


def export_mp4(frames: List[np.ndarray], path: str, fps: float = 25.0) -> None:
    """Képkockák írása MP4 videóba (cv2.VideoWriter).

    Először az 'avc1' (H.264) kodeket próbáljuk, ami szélesebb körben
    lejátszható (böngészők, modern médialejátszók).  Ha ez nem nyílik meg
    (pl. a rendszeren nincs H.264 encoder), visszaesünk az 'mp4v'-re.
    """
    if cv2 is None:
        raise RuntimeError("Az OpenCV (cv2) nincs telepítve.")
    if not frames:
        raise ValueError("Nincs exportálható képkocka.")
    h, w = frames[0].shape[:2]

    writer: Optional[cv2.VideoWriter] = None
    for codec in ("avc1", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        candidate = cv2.VideoWriter(path, fourcc, float(fps), (w, h))
        if candidate.isOpened():
            writer = candidate
            break
        candidate.release()

    if writer is None:
        raise RuntimeError(f"Nem sikerült megnyitni a videó-írót: {path}")
    try:
        for f in frames:
            writer.write(f)
    finally:
        writer.release()


def export_gif(frames: List[np.ndarray], path: str, fps: float = 15.0) -> None:
    """Képkockák írása animált GIF-be (Pillow)."""
    if not _HAS_PIL:
        raise RuntimeError(
            "A Pillow könyvtár nincs telepítve.\n"
            "Telepítsd: pip install Pillow")
    if not frames:
        raise ValueError("Nincs exportálható képkocka.")
    duration_ms = int(round(1000.0 / max(fps, 1.0)))
    pil_frames  = []
    for f in frames:
        rgb = f[:, :, ::-1]          # BGR → RGB
        pil_frames.append(_PilImage.fromarray(rgb))
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        loop=0,
        duration=duration_ms,
        optimize=False,
    )


def export_png_sequence(frames: List[np.ndarray], folder: str) -> None:
    """Képkockák mentése PNG sorozatként a megadott mappába."""
    if cv2 is None:
        raise RuntimeError("Az OpenCV (cv2) nincs telepítve.")
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    digits = len(str(len(frames)))
    for i, f in enumerate(frames):
        fname = out / f"frame_{i:0{digits}d}.png"
        cv2.imwrite(str(fname), f)


# ════════════════════════════════════════════════════════════════════════════
#  Delaunay háromszög morph
# ════════════════════════════════════════════════════════════════════════════

def _add_boundary_points(
    w: int, h: int,
    pts_a: List[Tuple], pts_b: List[Tuple],
) -> Tuple[List[Tuple], List[Tuple]]:
    """Sarok- és élközép-pontok hozzáadása a teljes képterület lefedéséhez."""
    boundary = [
        (0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1),
        (w // 2, 0), (w // 2, h - 1),
        (0, h // 2), (w - 1, h // 2),
    ]
    return (list(pts_a) + boundary, list(pts_b) + boundary)


def _morph_triangle(
    img1:    np.ndarray,
    img2:    np.ndarray,
    out:     np.ndarray,
    t1:      List[Tuple[int, int]],
    t2:      List[Tuple[int, int]],
    ti:      List[Tuple[int, int]],
    alpha:   float,
) -> None:
    """
    Két háromszög (t1 képből és t2 képből) affin-transzformálása a ti
    köztes háromszögbe, majd alpha-keverés az out képre.
    """
    h_out, w_out = out.shape[:2]

    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    ri = cv2.boundingRect(np.float32([ti]))

    # Biztonsági ellenőrzések
    if (ri[2] <= 0 or ri[3] <= 0 or r1[2] <= 0 or r1[3] <= 0
            or r2[2] <= 0 or r2[3] <= 0):
        return
    if (ri[0] >= w_out or ri[1] >= h_out
            or ri[0] + ri[2] <= 0 or ri[1] + ri[3] <= 0):
        return

    # Eltolt koordináták a bounding-rect origóhoz képest
    t1_off = np.float32([[p[0] - r1[0], p[1] - r1[1]] for p in t1])
    t2_off = np.float32([[p[0] - r2[0], p[1] - r2[1]] for p in t2])
    ti_off = np.float32([[p[0] - ri[0], p[1] - ri[1]] for p in ti])

    try:
        M1 = cv2.getAffineTransform(t1_off, ti_off)
        M2 = cv2.getAffineTransform(t2_off, ti_off)
    except cv2.error:
        return

    # Képrészletek kimetszése
    y1s, y1e = max(0, r1[1]), min(img1.shape[0], r1[1] + r1[3])
    x1s, x1e = max(0, r1[0]), min(img1.shape[1], r1[0] + r1[2])
    y2s, y2e = max(0, r2[1]), min(img2.shape[0], r2[1] + r2[3])
    x2s, x2e = max(0, r2[0]), min(img2.shape[1], r2[0] + r2[2])
    if y1s >= y1e or x1s >= x1e or y2s >= y2e or x2s >= x2e:
        return

    patch1 = img1[y1s:y1e, x1s:x1e].astype(np.float32)
    patch2 = img2[y2s:y2e, x2s:x2e].astype(np.float32)

    # BORDER_CONSTANT (fekete) – BORDER_REFLECT_101 tükörképes műterméket okoz
    w1 = cv2.warpAffine(patch1, M1, (ri[2], ri[3]),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0))
    w2 = cv2.warpAffine(patch2, M2, (ri[2], ri[3]),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0))

    # Háromszög-maszk
    mask = np.zeros((ri[3], ri[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(ti_off), (1.0, 1.0, 1.0),
                       lineType=cv2.LINE_AA)

    blended = (1.0 - alpha) * w1 + alpha * w2

    # Kimenet ROI
    ys = max(0, ri[1]); ye = min(h_out, ri[1] + ri[3])
    xs = max(0, ri[0]); xe = min(w_out, ri[0] + ri[2])
    dy = ye - ys; dx = xe - xs
    if dy <= 0 or dx <= 0:
        return

    # A warpolt patch és maszk megfelelő szelete
    patch_dy = ye - ri[1]; patch_dx = xe - ri[0]
    ps_y = ri[1] - ys if ri[1] < 0 else 0   # (általában 0)
    ps_x = ri[0] - xs if ri[0] < 0 else 0

    b_slice  = blended[ps_y:ps_y + dy, ps_x:ps_x + dx]
    m_slice  = mask[ps_y:ps_y + dy, ps_x:ps_x + dx]
    roi      = out[ys:ye, xs:xe]
    if b_slice.shape[:2] == roi.shape[:2] == m_slice.shape[:2]:
        roi[:] = roi * (1.0 - m_slice) + b_slice * m_slice


def generate_morph_frames_triangle(
    img_a:     np.ndarray,
    img_b:     np.ndarray,
    pts_a:     List[Tuple[float, float]],
    pts_b:     List[Tuple[float, float]],
    n_frames:  int  = 40,
    easing:    str  = "S-görbe",
    ping_pong: bool = False,
) -> List[np.ndarray]:
    """
    Klasszikus Delaunay-alapú háromszög-morph (Beier–Neely stílusú).
    Mindkét képet affin-transzformációkkal warplja a köztes pontokhoz,
    majd alpha-keveréssel ötvözi őket – valódi morph hatás.
    """
    if cv2 is None:
        raise RuntimeError("Az OpenCV (cv2) nincs telepítve.")
    if len(pts_a) < 3 or len(pts_b) < 3:
        raise ValueError(
            "Delaunay morphhoz legalább 3 pontpár szükséges.\n"
            "Adj hozzá több anchor pontot, vagy válassz más módszert!")

    h, w = img_b.shape[:2]

    # Pontok int-té konvertálva + határpontok hozzáadva
    pa = [(int(round(x)), int(round(y))) for x, y in pts_a]
    pb = [(int(round(x)), int(round(y))) for x, y in pts_b]
    pa, pb = _add_boundary_points(w, h, pa, pb)

    # Átlagpozíciók → erre számítjuk a Delaunay-t
    pm = [((pa[i][0] + pb[i][0]) // 2, (pa[i][1] + pb[i][1]) // 2)
          for i in range(len(pa))]

    # Delaunay-triangulálás
    # A téglalapot 2 pixellel bővítjük minden irányban, hogy a kerekítés
    # és határpont-eltolás miatt kívülre eső pontok ne okozzanak
    # cv::Subdiv2D::locate hibát (-211: out of range).
    _M = 2
    subdiv = cv2.Subdiv2D((-_M, -_M, w + _M, h + _M))
    for p in pm:
        # Pontokat is szorítjuk a kiterjesztett tartományba (biztonság kedvéért)
        px = float(max(-_M + 1, min(w + _M - 1, p[0])))
        py = float(max(-_M + 1, min(h + _M - 1, p[1])))
        subdiv.insert((px, py))
    tri_list = subdiv.getTriangleList().astype(np.int32)

    # Koordináta → index leképezés
    pt_idx = {(p[0], p[1]): i for i, p in enumerate(pm)}
    triangles: List[List[int]] = []
    for tri in tri_list:
        verts = [(tri[0], tri[1]), (tri[2], tri[3]), (tri[4], tri[5])]
        if all(0 <= v[0] < w and 0 <= v[1] < h for v in verts):
            idxs = [pt_idx.get(v) for v in verts]
            if None not in idxs:
                triangles.append(idxs)   # type: ignore[arg-type]

    func   = EASING_FUNCS.get(easing, _easing_linear)
    n      = max(n_frames, 2)
    frames: List[np.ndarray] = []

    for i in range(n):
        t_raw = i / (n - 1)
        t     = func(t_raw)

        # Köztes pontok
        pt = [(int(pa[j][0] * (1 - t) + pb[j][0] * t),
               int(pa[j][1] * (1 - t) + pb[j][1] * t))
              for j in range(len(pa))]

        out = np.zeros((h, w, 3), dtype=np.float32)
        for tri_idx in triangles:
            _morph_triangle(
                img_a, img_b, out,
                [pa[k] for k in tri_idx],
                [pb[k] for k in tri_idx],
                [pt[k] for k in tri_idx],
                alpha=t,
            )
        frames.append(np.clip(out, 0, 255).astype(np.uint8))

    if ping_pong:
        frames = frames + frames[-2:0:-1]
    return frames


# ════════════════════════════════════════════════════════════════════════════
#  Optikai folyam morph
# ════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════
#  TPS (Thin Plate Spline) morph
# ════════════════════════════════════════════════════════════════════════════

def generate_morph_frames_tps(
    img_a:     np.ndarray,
    img_b:     np.ndarray,
    pts_a:     List[Tuple[float, float]],
    pts_b:     List[Tuple[float, float]],
    n_frames:  int   = 40,
    easing:    str   = "S-görbe",
    ping_pong: bool  = False,
    smoothing: float = 0.0,
    scale:     float = 1.0,   # 0.5 = fél felbontáson számolja a map-et (gyorsabb)
) -> List[np.ndarray]:
    """
    Thin Plate Spline alapú morph.

    A TPS globálisan sima deformációs mezőt számol, ezért nincs háromszög-határ
    és természetesebb az egyenes vonalak (kősorok, ablakkeretek) morfja.

    Algoritmus:
      • Kontrollpont-párokat (pa→pb) t-vel interpolálva kapjuk a köztes pi pozíciókat.
      • Inverz TPS: pi → pa  és  pi → pb  (output-pixelből forrás-pixel).
      • cv2.remap a két képre, majd alpha-keverés.
      • Határrögzítők (sarok + élközép pontok, identity mapping) megelőzik
        a konvex hull szélén való kilengést.

    Paraméterek:
      smoothing : 0.0 = pontos interpoláció (átmegy minden kontrollponton);
                  >0  = simítás (outlier-tűrés, de elveszti az egzaktságot)
      scale     : remap-térkép számítási felbontása (0.5 = gyorsabb, kicsit durvább)
    """
    try:
        from scipy.interpolate import RBFInterpolator
    except ImportError:
        raise RuntimeError(
            "A TPS morphhoz a scipy csomag szükséges.\n"
            "Telepítés: pip install scipy")

    if cv2 is None:
        raise RuntimeError("Az OpenCV (cv2) nincs telepítve.")
    if len(pts_a) < 4 or len(pts_b) < 4:
        raise ValueError(
            "TPS morphhoz legalább 4 pontpár szükséges!\n"
            "Adj hozzá több anchor pontot, vagy válassz más módszert.")

    h, w = img_b.shape[:2]

    # ── Kontrollpontok float64 tömbként ──────────────────────────────────────
    pa_np = np.array(pts_a, dtype=np.float64)
    pb_np = np.array(pts_b, dtype=np.float64)

    # ── Határrögzítők (sarokpontok + élközéppontok, identity mapping) ────────
    # Ezek megakadályozzák hogy a TPS a képhatáron kívül kilengjen.
    border = np.array([
        [0,       0],     [w * .25, 0],     [w * .5, 0],     [w * .75, 0],     [w - 1, 0],
        [0,       h*.25], [w - 1,   h*.25],
        [0,       h*.5],  [w - 1,   h*.5],
        [0,       h*.75], [w - 1,   h*.75],
        [0,       h - 1], [w * .25, h - 1], [w * .5, h - 1], [w * .75, h - 1], [w - 1, h - 1],
    ], dtype=np.float64)

    pa_full = np.vstack([pa_np, border])
    pb_full = np.vstack([pb_np, border])

    # ── Sűrű értékelési rács (opcionálisan kisebb felbontáson) ───────────────
    s    = float(np.clip(scale, 0.25, 1.0))
    sw   = max(4, int(round(w * s)))
    sh   = max(4, int(round(h * s)))
    gy, gx = np.mgrid[0:sh, 0:sw]
    # Visszaskálázott koordináták az eredeti képtérbe
    gx_f = (gx.astype(np.float64) / (sw - 1)) * (w - 1)
    gy_f = (gy.astype(np.float64) / (sh - 1)) * (h - 1)
    grid = np.column_stack([gx_f.ravel(), gy_f.ravel()])  # (sh*sw, 2)

    func   = EASING_FUNCS.get(easing, _easing_linear)
    n      = max(n_frames, 2)
    frames: List[np.ndarray] = []

    for i in range(n):
        t_raw = i / (n - 1)
        t     = func(t_raw)

        # Köztes pozíciók (lineárisan interpolálva a kontrollpontokon)
        pi_full = pa_full * (1.0 - t) + pb_full * t

        # Inverz TPS: pi → pa  (melyik A-pixelből jön az output pixel)
        rbf_a = RBFInterpolator(pi_full, pa_full,
                                kernel='thin_plate_spline', smoothing=smoothing)
        # Inverz TPS: pi → pb
        rbf_b = RBFInterpolator(pi_full, pb_full,
                                kernel='thin_plate_spline', smoothing=smoothing)

        src_a = rbf_a(grid)   # (sh*sw, 2)
        src_b = rbf_b(grid)

        # Remap-térképek (small→full felbontás, ha scale < 1)
        def _make_maps(src):
            mx = src[:, 0].reshape(sh, sw).astype(np.float32)
            my = src[:, 1].reshape(sh, sw).astype(np.float32)
            if s < 1.0:
                mx = cv2.resize(mx, (w, h), interpolation=cv2.INTER_LINEAR)
                my = cv2.resize(my, (w, h), interpolation=cv2.INTER_LINEAR)
            return mx, my

        map_xa, map_ya = _make_maps(src_a)
        map_xb, map_yb = _make_maps(src_b)

        warped_a = cv2.remap(img_a, map_xa, map_ya,
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        warped_b = cv2.remap(img_b, map_xb, map_yb,
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        frame = cv2.addWeighted(warped_a, 1.0 - t, warped_b, t, 0.0)
        frames.append(frame)

    if ping_pong:
        frames = frames + frames[-2:0:-1]
    return frames


# ════════════════════════════════════════════════════════════════════════════
#  Optikai folyam morph
# ════════════════════════════════════════════════════════════════════════════

def generate_morph_frames_flow(
    img_a:      np.ndarray,
    img_b:      np.ndarray,
    n_frames:   int   = 40,
    easing:     str   = "S-görbe",
    ping_pong:  bool  = False,
    pyr_scale:  float = 0.5,
    levels:     int   = 4,
    winsize:    int   = 21,
    iterations: int   = 4,
    poly_n:     int   = 7,
    poly_sigma: float = 1.5,
) -> List[np.ndarray]:
    """
    Optikai folyam alapú morph (Farneback dense flow).
    Mindkét képet a flow-mező alapján deformálja a köztes állapot felé,
    majd alpha-keveréssel ötvözi – pontpár nélkül is organikus mozgást ad.
    """
    if cv2 is None:
        raise RuntimeError("Az OpenCV (cv2) nincs telepítve.")

    h, w = img_a.shape[:2]
    # Célkép mérethez igazítás ha szükséges
    if img_b.shape[:2] != (h, w):
        img_b = cv2.resize(img_b, (w, h))

    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

    # A→B és B→A flow
    farneback_kw = dict(
        pyr_scale=pyr_scale, levels=levels, winsize=winsize,
        iterations=iterations, poly_n=poly_n, poly_sigma=poly_sigma,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
    )
    flow_ab = cv2.calcOpticalFlowFarneback(gray_a, gray_b, None, **farneback_kw)
    flow_ba = cv2.calcOpticalFlowFarneback(gray_b, gray_a, None, **farneback_kw)

    # Pixel-rács
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                          np.arange(h, dtype=np.float32))

    func   = EASING_FUNCS.get(easing, _easing_linear)
    n      = max(n_frames, 2)
    frames: List[np.ndarray] = []

    for i in range(n):
        t_raw = i / (n - 1)
        t     = func(t_raw)

        # A képet t irányban deformáljuk B felé
        map_ax = gx + flow_ab[..., 0] * t
        map_ay = gy + flow_ab[..., 1] * t
        wa = cv2.remap(img_a, map_ax, map_ay,
                       cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        # B képet (1-t) irányban deformáljuk A felé
        map_bx = gx + flow_ba[..., 0] * (1.0 - t)
        map_by = gy + flow_ba[..., 1] * (1.0 - t)
        wb = cv2.remap(img_b, map_bx, map_by,
                       cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        frame = cv2.addWeighted(wa, 1.0 - t, wb, t, 0.0)
        frames.append(frame)

    if ping_pong:
        frames = frames + frames[-2:0:-1]
    return frames


# ════════════════════════════════════════════════════════════════════════════
#  MatchOverviewCanvas  –  az összes automatikus egyezés megjelenítése
# ════════════════════════════════════════════════════════════════════════════

class MatchOverviewCanvas(QWidget):
    """QWidget alapú – paintEvent() rajzol, nem setPixmap()."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._image_a = self._image_b = None
        self._raw_a   = np.empty((0, 2))
        self._raw_b   = np.empty((0, 2))
        self._mask    = np.empty((0,), dtype=bool)
        self._limit   = 150

    def sizeHint(self) -> QSize:
        return QSize(600, 200)

    def set_images(self, a, b) -> None:
        self._image_a, self._image_b = a, b
        self.update()

    def set_matches(self, ra, rb, m, limit: int = 150) -> None:
        self._raw_a, self._raw_b, self._mask, self._limit = ra, rb, m, limit
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#14181d"))
        painter.setPen(QPen(QColor("#343b45"), 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))

        if self._image_a is None or self._image_b is None:
            painter.setPen(QPen(QColor("#4a5260"), 1))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "Nincs adat az összehasonlításhoz.")
            return

        w_half = max(1, (self.width() - 30) // 2)
        h_max  = max(1, self.height() - 10)

        pix_a = bgr_to_qpixmap(self._image_a).scaled(
            w_half, h_max, Qt.AspectRatioMode.KeepAspectRatio)
        pix_b = bgr_to_qpixmap(self._image_b).scaled(
            w_half, h_max, Qt.AspectRatioMode.KeepAspectRatio)

        ra = QRectF(10, 5, pix_a.width(), pix_a.height())
        rb = QRectF(20 + w_half, 5, pix_b.width(), pix_b.height())

        painter.drawPixmap(int(ra.left()), int(ra.top()), pix_a)
        painter.drawPixmap(int(rb.left()), int(rb.top()), pix_b)

        if len(self._raw_a) > 0 and len(self._raw_a) == len(self._raw_b):
            step = max(1, len(self._raw_a) // self._limit)
            for i in range(0, len(self._raw_a), step):
                inlier = self._mask[i] if i < len(self._mask) else True
                color  = QColor(0, 255, 0, 140) if inlier else QColor(255, 0, 0, 70)
                painter.setPen(QPen(color, 1))
                pa_x = ra.left() + (self._raw_a[i][0] / self._image_a.shape[1]) * ra.width()
                pa_y = ra.top()  + (self._raw_a[i][1] / self._image_a.shape[0]) * ra.height()
                pb_x = rb.left() + (self._raw_b[i][0] / self._image_b.shape[1]) * rb.width()
                pb_y = rb.top()  + (self._raw_b[i][1] / self._image_b.shape[0]) * rb.height()
                painter.drawLine(int(pa_x), int(pa_y), int(pb_x), int(pb_y))


# ════════════════════════════════════════════════════════════════════════════
#  AlignmentPreviewCanvas
# ════════════════════════════════════════════════════════════════════════════

class AlignmentPreviewCanvas(QWidget):
    """QWidget alapú – paintEvent() rajzol, nem setPixmap()."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._img        = None
        self._view_mode  = tr("Overlay")
        self._all_imgs: Dict[str, Any] = {}   # mindhárom mód képe tárolva

    def sizeHint(self) -> QSize:
        return QSize(600, 400)

    def set_images(self, target, warped, overlay) -> None:
        """Mindhárom nézet képét eltárolja; az aktív módot azonnal mutatja."""
        self._all_imgs = {tr("Célkép"): target, tr("Warpolt A"): warped, tr("Overlay"): overlay}
        self._img = self._all_imgs.get(self._view_mode, overlay)
        self.update()

    def set_view_mode(self, mode: str) -> None:
        """Nézetváltás – ha már vannak képek, azonnal frissül."""
        self._view_mode = mode
        if self._all_imgs:
            self._img = self._all_imgs.get(mode)
            self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#14181d"))
        painter.setPen(QPen(QColor("#343b45"), 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))

        if self._img is None:
            painter.setPen(QPen(QColor("#4a5260"), 1))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "Nincs előnézet.")
            return

        pix = bgr_to_qpixmap(self._img).scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = (self.width()  - pix.width())  // 2
        y = (self.height() - pix.height()) // 2
        painter.drawPixmap(x, y, pix)


# ════════════════════════════════════════════════════════════════════════════
#  Fülek
# ════════════════════════════════════════════════════════════════════════════

class AutomaticModeTab(QWidget):
    """
    Automatikus illesztés fül.
    Minden backendhez saját paraméterpanel (QStackedWidget),
    minden widgetnek magyar nyelvű tooltip.
    """
    request_auto_match = pyqtSignal()

    # ── Segéd: sor hozzáadása tooltip-pel ───────────────────────────────────

    @staticmethod
    def _add_row(form: QFormLayout, label: str, widget, tip: str) -> None:
        widget.setToolTip(tip)
        lbl = QLabel(label)
        lbl.setToolTip(tip)
        form.addRow(lbl, widget)

    # ── Per-backend panelek ──────────────────────────────────────────────────

    def _panel_superpoint(self) -> QWidget:
        w    = QWidget()
        form = QFormLayout(w)
        form.setContentsMargins(0, 0, 0, 0)

        self.sp_max_kp = QSpinBox()
        self.sp_max_kp.setRange(128, 8192)
        self.sp_max_kp.setValue(cfg("match.superpoint.max_keypoints", 4096))
        self._add_row(form, "Max. kulcspontok:", self.sp_max_kp,
            "A SuperPoint detektor által megtartott kulcspontok maximális száma.\n"
            "Több pont → pontosabb illesztés, de lassabb futás és több GPU-memória.\n\n"
            "Forrás: a cvg/LightGlue GitHub-repo hivatalos demói és benchmarkjai\n"
            "4096-ot használnak alapértelmezettként ('max_num_keypoints: 4096').\n"
            "Ajánlott: 2048–4096. Tartomány: 128–8192.")

        self.sp_det_thresh = QDoubleSpinBox()
        self.sp_det_thresh.setRange(0.0001, 0.99)
        self.sp_det_thresh.setSingleStep(0.001)
        self.sp_det_thresh.setDecimals(4)
        self.sp_det_thresh.setValue(cfg("match.superpoint.detection_threshold", 0.005))
        self._add_row(form, "Detekciós küszöb:", self.sp_det_thresh,
            "A SuperPoint detektor érzékenységi küszöbe (detection_threshold).\n"
            "Alacsonyabb érték → több pont detektálódik, beleértve a gyengébbeket is.\n"
            "Magasabb érték → csak az erős, megbízható sarokpontok maradnak meg.\n\n"
            "Forrás: SuperPoint hivatalos implementáció (Magic Leap / rpautrat/SuperPoint),\n"
            "Hugging Face transformers, cvg/LightGlue – mindegyik 0.005-öt használ.\n"
            "Ajánlott: 0.001–0.010.")

        self.sp_nms_radius = QSpinBox()
        self.sp_nms_radius.setRange(1, 20)
        self.sp_nms_radius.setValue(cfg("match.superpoint.nms_radius", 4))
        self._add_row(form, "NMS sugár (px):", self.sp_nms_radius,
            "Non-Maximum Suppression sugár pixelben (nms_radius).\n"
            "Megakadályozza, hogy két kulcspont egymáshoz túl közel legyen.\n"
            "Nagyobb érték → ritkább, de egyenletesebb eloszlású pontok.\n"
            "Kisebb érték → sűrűbb pontok, de lehetséges átfedés.\n\n"
            "Forrás: SuperPoint eredeti cikk (DeTone et al. 2018) és a cvg/LightGlue\n"
            "repo – mindkettő 4 px-t ajánl alapértelmezettként.\n"
            "Ajánlott: 3–6.")

        self.sp_match_thresh = QDoubleSpinBox()
        self.sp_match_thresh.setRange(0.01, 1.0)
        self.sp_match_thresh.setSingleStep(0.01)
        self.sp_match_thresh.setDecimals(3)
        self.sp_match_thresh.setValue(cfg("match.superpoint.match_threshold", 0.10))
        self._add_row(form, "Match küszöb (LG):", self.sp_match_thresh,
            "A LightGlue párosító szűrési küszöbe (filter_threshold).\n"
            "Alacsonyabb → szigorúbb szűrés: kevesebb, de pontosabb egyezés.\n"
            "Magasabb → több egyezés, de több hibás pár is átcsúszik a szűrőn.\n\n"
            "Forrás: cvg/LightGlue – lightglue/lightglue.py, 'filter_threshold: 0.1'\n"
            "Ez a könyvtár hivatalos alapértelmezettje.\n"
            "Ajánlott: 0.05–0.20.")

        self.sp_depth_conf = QDoubleSpinBox()
        self.sp_depth_conf.setRange(0.5, 1.0)
        self.sp_depth_conf.setSingleStep(0.01)
        self.sp_depth_conf.setDecimals(2)
        self.sp_depth_conf.setValue(cfg("match.superpoint.depth_confidence", 0.95))
        self._add_row(form, "Mélység-konfidencia:", self.sp_depth_conf,
            "A LightGlue korai leállás küszöbe mélység irányban (depth_confidence).\n"
            "Ha az egyezések elég megbízhatók, a modell korábban megáll → gyorsabb.\n"
            "1.0 = kikapcsolva (a modell végigfut minden rétegen).\n\n"
            "Forrás: LightGlue ICCV 2023 cikk (Lindenberger et al.) és a cvg/LightGlue\n"
            "repo – alapértelmezett érték: 0.95.\n"
            "Ajánlott: 0.90–0.98.")

        self.sp_width_conf = QDoubleSpinBox()
        self.sp_width_conf.setRange(0.5, 1.0)
        self.sp_width_conf.setSingleStep(0.01)
        self.sp_width_conf.setDecimals(2)
        self.sp_width_conf.setValue(cfg("match.superpoint.width_confidence", 0.99))
        self._add_row(form, "Szélesség-konfidencia:", self.sp_width_conf,
            "A LightGlue korai leállás küszöbe szélesség irányban (width_confidence).\n"
            "Ha az összes pont kezelve van elég biztosan, a modell megáll → gyorsabb.\n"
            "1.0 = kikapcsolva.\n\n"
            "Forrás: LightGlue ICCV 2023 cikk és a cvg/LightGlue repo –\n"
            "alapértelmezett érték: 0.99.\n"
            "Ajánlott: 0.95–1.0.")

        return w

    def _panel_disk(self) -> QWidget:
        w    = QWidget()
        form = QFormLayout(w)
        form.setContentsMargins(0, 0, 0, 0)

        self.dk_max_kp = QSpinBox()
        self.dk_max_kp.setRange(128, 8192)
        self.dk_max_kp.setValue(cfg("match.disk.max_keypoints", 5000))
        self._add_row(form, "Max. kulcspontok:", self.dk_max_kp,
            "A DISK detektor által megtartott kulcspontok maximális száma.\n"
            "A DISK ismétlődő mintákon (ablaksorok, csempék, kövezet) jobban\n"
            "teljesít mint a SuperPoint, mert ezekre van betanítva.\n\n"
            "Forrás: Parskatt/DeDoDe és a kornia.feature.DISK implementációk\n"
            "~5000–10000 kulcspontot használnak alapértelmezettként, mert\n"
            "a DISK sűrűbben mintavételez mint a SuperPoint.\n"
            "Ajánlott: 3000–8000.")

        self.dk_match_thresh = QDoubleSpinBox()
        self.dk_match_thresh.setRange(0.01, 1.0)
        self.dk_match_thresh.setSingleStep(0.01)
        self.dk_match_thresh.setDecimals(3)
        self.dk_match_thresh.setValue(cfg("match.disk.match_threshold", 0.10))
        self._add_row(form, "Match küszöb (LG):", self.dk_match_thresh,
            "A LightGlue párosító szűrési küszöbe (filter_threshold).\n"
            "Alacsonyabb → szigorúbb szűrés: kevesebb, de pontosabb egyezés.\n"
            "Magasabb → több egyezés, de több hibás pár is átcsúszik a szűrőn.\n\n"
            "Forrás: cvg/LightGlue – lightglue/lightglue.py, 'filter_threshold: 0.1'\n"
            "Ez a könyvtár hivatalos alapértelmezettje (SuperPoint és DISK esetén egyaránt).\n"
            "Ajánlott: 0.05–0.20.")

        note = QLabel(
            "<i>A DISK ugyanazt a LightGlue matcher-t használja mint a SuperPoint,<br>"
            "de különböző jellemzőteret detektál – ismétlődő mintákon jobb lehet.</i>")
        note.setWordWrap(True)
        note.setStyleSheet("color:#8a9ab0; font-size:11px; margin-top:6px;")
        form.addRow(note)
        return w

    def _panel_loftr(self) -> QWidget:
        w    = QWidget()
        form = QFormLayout(w)
        form.setContentsMargins(0, 0, 0, 0)

        self.lf_pretrained = QComboBox()
        self.lf_pretrained.addItems([tr("outdoor"), tr("indoor")])
        self.lf_pretrained.setCurrentText(cfg("match.loftr.pretrained", tr("outdoor")))
        self._add_row(form, "Előtanított modell:", self.lf_pretrained,
            "Az előre tanított LoFTR modell típusa.\n\n"
            "'outdoor' (kültéri): épületek, utcák, terek, tájképek, műemlékek.\n"
            "'indoor' (beltéri): szobák, folyosók, termek, belső terek.\n\n"
            "Válaszd a képeid tartalmához leginkább illőt.")

        self.lf_conf_thresh = QDoubleSpinBox()
        self.lf_conf_thresh.setRange(0.0, 1.0)
        self.lf_conf_thresh.setSingleStep(0.05)
        self.lf_conf_thresh.setDecimals(2)
        self.lf_conf_thresh.setValue(cfg("match.loftr.conf_threshold", 0.50))
        self._add_row(form, "Konfidencia küszöb:", self.lf_conf_thresh,
            "Az egyezés megbízhatósági küszöbe.\n"
            "Csak az ennél magasabb konfidenciájú egyezések maradnak meg.\n"
            "Alacsonyabb → több egyezés, de több zajpont is.\n"
            "Magasabb → kevesebb, de pontosabb egyezés.\n"
            "Ajánlott: 0.30–0.70.")

        note = QLabel(
            "<i>A LoFTR detektor nélküli (detector-free) illesztő: sűrű, pixelszintű<br>"
            "egyezéseket keres. Különösen jó textúraszegény vagy erősen deformált<br>"
            "képpárokon (pl. historikus + modern fotó). CPU-n lassabb mint a SIFT.</i>")
        note.setWordWrap(True)
        note.setStyleSheet("color:#8a9ab0; font-size:11px; margin-top:6px;")
        form.addRow(note)
        return w

    def _panel_sift(self) -> QWidget:
        w    = QWidget()
        form = QFormLayout(w)
        form.setContentsMargins(0, 0, 0, 0)

        self.si_max_kp = QSpinBox()
        self.si_max_kp.setRange(0, 8192)
        self.si_max_kp.setValue(cfg("match.sift.max_keypoints", 0))
        self._add_row(form, "Max. kulcspontok:", self.si_max_kp,
            "A SIFT által detektált kulcspontok maximális száma (nfeatures).\n"
            "0 = korlátlan (minden detektált pont megmarad).\n\n"
            "Forrás: Lowe IJCV 2004 eredeti cikkje és az OpenCV SIFT dokumentáció –\n"
            "mindkettő 0-t (korlátlan) ad meg alapértelmezettként.\n"
            "Ha sok a felesleges pont, a RANSAC szűrő eltávolítja őket.\n"
            "Ajánlott: 0 (korlátlan) vagy 2000–5000 ha lassú a feldolgozás.")

        self.si_octave_layers = QSpinBox()
        self.si_octave_layers.setRange(1, 10)
        self.si_octave_layers.setValue(cfg("match.sift.octave_layers", 3))
        self._add_row(form, "Oktáv-rétegek:", self.si_octave_layers,
            "Az oktávonkénti rétegek száma a Gauss-skálatérben (nOctaveLayers).\n"
            "Több réteg → finomabb méretarány-invariancia, de lassabb futás.\n\n"
            "Forrás: Lowe IJCV 2004 – 3 az eredeti értéke ('s=3 gives good results').\n"
            "Az OpenCV alapértelmezettje is 3. Általában nem érdemes változtatni.")

        self.si_contrast = QDoubleSpinBox()
        self.si_contrast.setRange(0.001, 0.5)
        self.si_contrast.setSingleStep(0.001)
        self.si_contrast.setDecimals(3)
        self.si_contrast.setValue(cfg("match.sift.contrast_threshold", 0.04))
        self._add_row(form, "Kontraszt küszöb:", self.si_contrast,
            "Alacsony kontrasztú kulcspontok szűrési küszöbe (contrastThreshold).\n"
            "Alacsonyabb → több pont, beleértve a gyenge kontrasztú területeket.\n"
            "Magasabb → csak erős kontrasztú, megbízható pontok maradnak.\n\n"
            "Forrás: Lowe IJCV 2004 – 0.04 az eredeti ajánlott érték.\n"
            "Az OpenCV alapértelmezettje szintén 0.04.\n"
            "Ajánlott: 0.02–0.08.")

        self.si_edge = QDoubleSpinBox()
        self.si_edge.setRange(1.0, 50.0)
        self.si_edge.setSingleStep(1.0)
        self.si_edge.setDecimals(1)
        self.si_edge.setValue(cfg("match.sift.edge_threshold", 10.0))
        self._add_row(form, "Él küszöb:", self.si_edge,
            "Az élszűrő küszöbe (edgeThreshold).\n"
            "Az élek mentén lévő instabil kulcspontokat szűri ki.\n"
            "Magasabb → kevesebb él-jellegű pont törlése (több pont marad).\n"
            "Alacsonyabb → szigorúbb élszűrés (kevesebb, de stabilabb pont).\n\n"
            "Forrás: Lowe IJCV 2004 – 10 az eredeti érték ('r=10').\n"
            "Az OpenCV alapértelmezettje szintén 10.\n"
            "Ajánlott: 5–20.")

        self.si_sigma = QDoubleSpinBox()
        self.si_sigma.setRange(0.5, 5.0)
        self.si_sigma.setSingleStep(0.1)
        self.si_sigma.setDecimals(1)
        self.si_sigma.setValue(cfg("match.sift.sigma", 1.6))
        self._add_row(form, "Gauss sigma:", self.si_sigma,
            "A Gauss-simítás sigma paramétere a skálatér alaplépésénél.\n"
            "Kisebb (1.2–1.4): kis képeken, kevésbé zajos képeken.\n"
            "Nagyobb (1.8–2.5): nagy, zajos képeken, erősebb simítás.\n\n"
            "Forrás: Lowe IJCV 2004 – 1.6 az eredeti értéke ('σ=1.6').\n"
            "Az OpenCV alapértelmezettje szintén 1.6. Általában nem kell változtatni.")

        self.si_ratio = QDoubleSpinBox()
        self.si_ratio.setRange(0.5, 0.99)
        self.si_ratio.setSingleStep(0.01)
        self.si_ratio.setDecimals(2)
        self.si_ratio.setValue(cfg("match.sift.ratio_threshold", 0.80))
        self._add_row(form, "Lowe arány küszöb:", self.si_ratio,
            "Lowe-féle arányküszöb a hamis egyezések szűrésére.\n"
            "Egy egyezés elfogadott, ha:\n"
            "  legjobb_távolság < arány × második_legjobb_távolság\n"
            "Alacsonyabb → szigorúbb: kevesebb, de pontosabb egyezés.\n"
            "Magasabb → több egyezés, de több hamis pár is.\n\n"
            "Forrás: Lowe IJCV 2004 – 0.8 az eredeti mért optimális érték\n"
            "('eliminates 90% of false matches while discarding less than 5%\n"
            "of correct matches'). Az OpenCV tutorial szintén 0.8-at ajánl.\n"
            "Ajánlott: 0.70–0.85.")

        note = QLabel(
            "<i>A SIFT (Scale-Invariant Feature Transform) klasszikus algoritmus –<br>"
            "GPU és PyTorch nélkül fut. Megbízható, mindig elérhető ha az OpenCV<br>"
            "telepítve van. Nagy képeken lassabb, mint a neurális módszerek.</i>")
        note.setWordWrap(True)
        note.setStyleSheet("color:#8a9ab0; font-size:11px; margin-top:6px;")
        form.addRow(note)
        return w

    def _panel_common(self) -> QWidget:
        """RANSAC + CPU kényszerítés – minden backenddel közös, mindig látható."""
        w    = QWidget()
        form = QFormLayout(w)
        form.setContentsMargins(0, 0, 0, 0)

        self.ransac_chk = QCheckBox("RANSAC szűrés bekapcsolva")
        self.ransac_chk.setChecked(True)
        self.ransac_chk.setToolTip(
            "RANSAC (Random Sample Consensus) geometriai szűrés.\n"
            "Automatikusan kiszűri a geometriailag hibás párokat (outliereket)\n"
            "a homográfia illesztés alapján.\n"
            "Erősen ajánlott, különösen sok pont és eltérő perspektíva esetén.")
        form.addRow(self.ransac_chk)

        self.ransac_reproj = QDoubleSpinBox()
        self.ransac_reproj.setRange(0.5, 20.0)
        self.ransac_reproj.setSingleStep(0.5)
        self.ransac_reproj.setDecimals(1)
        self.ransac_reproj.setValue(cfg("match.ransac.reproj_threshold", 3.0))
        self._add_row(form, "Reproj. küszöb (px):", self.ransac_reproj,
            "A RANSAC visszavetítési (reprojekciós) küszöb pixelben.\n"
            "Ha egy pontpár visszavetítési hibája nagyobb ennél, outliernek\n"
            "minősül és törlődik az illesztésből.\n"
            "Kisebb → szigorúbb szűrés (kevesebb pont, de pontosabb).\n"
            "Nagyobb → engedékenyebb szűrés (több pont marad).\n\n"
            "Forrás: OpenCV cv2.findHomography dokumentáció –\n"
            "alapértelmezett érték 3.0 px ('reproj_threshold=3').\n"
            "OpenCV feature homography tutorial szintén 3.0-t ajánl.\n"
            "Ajánlott: 2.0–5.0. Pixelszintű pontossághoz: 1.0–2.0.")

        self.cpu_chk = QCheckBox("CPU kényszerítése (debug / GPU nélkül)")
        self.cpu_chk.setChecked(False)
        self.cpu_chk.setToolTip(
            "Ha be van jelölve, az algoritmus GPU helyett CPU-n fut.\n"
            "Hasznos ha nincs kompatibilis GPU, vagy CUDA-hiba esetén.\n"
            "CUDA nélküli rendszeren automatikusan CPU-t használ,\n"
            "ezt akkor nem szükséges külön bejelölni.")
        form.addRow(self.cpu_chk)
        return w

    # ── Konstruktor + fő UI ──────────────────────────────────────────────────

    def __init__(self, settings: AppSettings, project: ProjectState) -> None:
        super().__init__()
        self.settings, self.project = settings, project
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(8)

        # ════════════════════════════════════════════════════════════════════
        # 1. ALGORITMUS + RANSAC – szorosan összetartoznak:
        #    a RANSAC az illesztési folyamat geometriai szűrő lépése
        # ════════════════════════════════════════════════════════════════════
        algo_box  = QGroupBox(tr("Algoritmus és geometriai szűrés"))
        algo_form = QFormLayout(algo_box)
        algo_form.setSpacing(6)

        self.matcher_combo = QComboBox()
        self.matcher_combo.addItems([
            "SuperPoint + LightGlue",
            "DISK + LightGlue",
            "LoFTR (kornia)",
            "SIFT (OpenCV)",
        ])
        self.matcher_combo.setToolTip(
            "Az automatikus pontillesztéshez használt algoritmus.\n\n"
            "SuperPoint + LightGlue:\n"
            "  Neurális, GPU-n gyors; legtöbb esetben a legjobb választás.\n\n"
            "DISK + LightGlue:\n"
            "  Ismétlődő mintákon (ablaksorok, kövezet, csempék) jobb.\n\n"
            "LoFTR (kornia):\n"
            "  Detektor nélküli; textúraszegény vagy erősen deformált képeken.\n\n"
            "SIFT (OpenCV):\n"
            "  Klasszikus, CPU-n fut, mindig elérhető; megbízható fallback.")
        algo_form.addRow(tr("Backend:"), self.matcher_combo)

        # Elválasztó vonal a RANSAC blokk előtt
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#3a4a5a;")
        algo_form.addRow(sep)

        self.ransac_chk = QCheckBox(tr("RANSAC geometriai szűrés"))
        self.ransac_chk.setChecked(True)
        self.ransac_chk.setToolTip(
            "RANSAC szűrés: automatikusan kizárja a geometriailag\n"
            "hibás pontpárokat (outliereket) a homográfia illesztés alapján.\n"
            "Erősen ajánlott, különösen sok pont és eltérő perspektíva esetén.")
        algo_form.addRow(self.ransac_chk)

        self.ransac_reproj = QDoubleSpinBox()
        self.ransac_reproj.setRange(0.5, 20.0)
        self.ransac_reproj.setSingleStep(0.5)
        self.ransac_reproj.setDecimals(1)
        self.ransac_reproj.setValue(cfg("match.ransac.reproj_threshold", 3.0))
        self._add_row(algo_form, tr("  Reproj. küszöb (px):"), self.ransac_reproj,
            "Visszavetítési küszöb pixelben.\n"
            "Kisebb → szigorúbb szűrés (kevesebb, de pontosabb egyezés).\n"
            "Nagyobb → engedékenyebb (több pont marad).\n"
            "Ajánlott: 2.0–5.0. Pixelszintű pontossághoz: 1.0–2.0.")

        root.addWidget(algo_box)

        # ════════════════════════════════════════════════════════════════════
        # 2. BACKEND PARAMÉTEREK – a kiválasztott algoritmustól függő beállítások
        # ════════════════════════════════════════════════════════════════════
        params_box  = QGroupBox(tr("Backend paraméterek"))
        params_vbox = QVBoxLayout(params_box)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._panel_superpoint())   # index 0
        self._stack.addWidget(self._panel_disk())         # index 1
        self._stack.addWidget(self._panel_loftr())        # index 2
        self._stack.addWidget(self._panel_sift())         # index 3

        params_vbox.addWidget(self._stack)
        root.addWidget(params_box)

        # ════════════════════════════════════════════════════════════════════
        # 3. HALADÓ BEÁLLÍTÁSOK – debug / speciális eset (CPU kényszerítés)
        # ════════════════════════════════════════════════════════════════════
        adv_box  = QGroupBox(tr("Haladó beállítások"))
        adv_vbox = QVBoxLayout(adv_box)
        adv_vbox.setContentsMargins(8, 4, 8, 4)

        self.cpu_chk = QCheckBox(tr("CPU kényszerítése  (GPU nélküli futtatás)"))
        self.cpu_chk.setChecked(False)
        self.cpu_chk.setToolTip(
            "Ha be van jelölve, az algoritmus GPU helyett CPU-n fut.\n"
            "Hasznos ha nincs kompatibilis GPU, vagy CUDA-hiba esetén.\n"
            "CUDA nélküli rendszeren automatikusan CPU-t használ –\n"
            "ilyenkor ezt nem szükséges bejelölni.")
        adv_vbox.addWidget(self.cpu_chk)
        root.addWidget(adv_box)

        # ════════════════════════════════════════════════════════════════════
        # 4. INDÍTÁS GOMB – legalsó, mindig látható
        # ════════════════════════════════════════════════════════════════════
        btn = QPushButton("▶  Automatikus illesztés indítása")
        btn.setStyleSheet(
            "QPushButton{background:#2e7d32;color:#eee;font-weight:bold;"
            "padding:10px;border-radius:5px;font-size:13px;}"
            "QPushButton:hover{background:#388e3c;}"
        )
        btn.setToolTip(
            "Elindítja az automatikus pontillesztést a kiválasztott backend\n"
            "és a fenti paraméterek alapján.\n"
            "A meglévő kézzel szerkesztett pontok felülíródnak!\n"
            "(Szerkesztés → Visszavonás menüvel visszaállítható.)")
        btn.clicked.connect(self.request_auto_match.emit)
        root.addWidget(btn)
        root.addStretch()

        # Combo → stack összekötése
        self.matcher_combo.currentIndexChanged.connect(self._stack.setCurrentIndex)

    # ── Paraméterek lekérdezése ──────────────────────────────────────────────

    def get_match_params(self) -> dict:
        """
        Az aktuálisan kiválasztott backend összes beállítását adja vissza
        egy dict-ben, amelyet közvetlenül a run_* backend-függvényeknek
        lehet **kwargs-ként átadni.

        Kulcsok:
            backend, use_ransac, reproj_threshold, force_cpu
            + backend-specifikus paraméterek (max_keypoints, stb.)
        """
        idx     = self.matcher_combo.currentIndex()
        backend = self.matcher_combo.currentText()

        params: dict = {
            "backend":          backend,
            "use_ransac":       self.ransac_chk.isChecked(),
            "reproj_threshold": self.ransac_reproj.value(),
            "force_cpu":        self.cpu_chk.isChecked(),
        }

        if idx == 0:    # SuperPoint + LightGlue
            params.update({
                "max_keypoints":       self.sp_max_kp.value(),
                "match_threshold":     self.sp_match_thresh.value(),
                "detection_threshold": self.sp_det_thresh.value(),
                "nms_radius":          self.sp_nms_radius.value(),
                "depth_confidence":    self.sp_depth_conf.value(),
                "width_confidence":    self.sp_width_conf.value(),
            })
        elif idx == 1:  # DISK + LightGlue
            params.update({
                "max_keypoints":   self.dk_max_kp.value(),
                "match_threshold": self.dk_match_thresh.value(),
            })
        elif idx == 2:  # LoFTR
            params.update({
                "pretrained":           self.lf_pretrained.currentText(),
                "confidence_threshold": self.lf_conf_thresh.value(),
            })
        elif idx == 3:  # SIFT
            params.update({
                "max_keypoints":      self.si_max_kp.value(),
                "n_octave_layers":    self.si_octave_layers.value(),
                "contrast_threshold": self.si_contrast.value(),
                "edge_threshold":     self.si_edge.value(),
                "sigma":              self.si_sigma.value(),
                "ratio_threshold":    self.si_ratio.value(),
            })

        return params

    def restore_params(self, d: Dict[str, Any]) -> None:
        """Visszaállítja az automata illesztés összes beállítását a mentett dict-ből."""
        backend = d.get("backend", "")
        if backend:
            idx = self.matcher_combo.findText(backend)
            if idx >= 0:
                self.matcher_combo.setCurrentIndex(idx)
        if "use_ransac" in d:
            self.ransac_chk.setChecked(bool(d["use_ransac"]))
        if "reproj_threshold" in d:
            self.ransac_reproj.setValue(float(d["reproj_threshold"]))
        if "force_cpu" in d:
            self.cpu_chk.setChecked(bool(d["force_cpu"]))
        # Backend-specifikus widgetek (csak ha a megfelelő fül aktív)
        idx = self.matcher_combo.currentIndex()
        if idx == 0:   # SuperPoint + LightGlue
            if "max_keypoints" in d:       self.sp_max_kp.setValue(int(d["max_keypoints"]))
            if "match_threshold" in d:     self.sp_match_thresh.setValue(float(d["match_threshold"]))
            if "detection_threshold" in d: self.sp_det_thresh.setValue(float(d["detection_threshold"]))
            if "nms_radius" in d:          self.sp_nms_radius.setValue(int(d["nms_radius"]))
            if "depth_confidence" in d:    self.sp_depth_conf.setValue(float(d["depth_confidence"]))
            if "width_confidence" in d:    self.sp_width_conf.setValue(float(d["width_confidence"]))
        elif idx == 1:  # DISK + LightGlue
            if "max_keypoints" in d:   self.dk_max_kp.setValue(int(d["max_keypoints"]))
            if "match_threshold" in d: self.dk_match_thresh.setValue(float(d["match_threshold"]))
        elif idx == 2:  # LoFTR
            if "pretrained" in d:
                pi = self.lf_pretrained.findText(str(d["pretrained"]))
                if pi >= 0: self.lf_pretrained.setCurrentIndex(pi)
            if "confidence_threshold" in d:
                self.lf_conf_thresh.setValue(float(d["confidence_threshold"]))
        elif idx == 3:  # SIFT
            if "max_keypoints" in d:      self.si_max_kp.setValue(int(d["max_keypoints"]))
            if "n_octave_layers" in d:    self.si_octave_layers.setValue(int(d["n_octave_layers"]))
            if "contrast_threshold" in d: self.si_contrast.setValue(float(d["contrast_threshold"]))
            if "edge_threshold" in d:     self.si_edge.setValue(float(d["edge_threshold"]))
            if "sigma" in d:              self.si_sigma.setValue(float(d["sigma"]))
            if "ratio_threshold" in d:    self.si_ratio.setValue(float(d["ratio_threshold"]))


# ── AdvancedEditorTab  (a PointEditorWidget-et használja) ───────────────────

class AdvancedEditorTab(QWidget):

    def __init__(self, settings: AppSettings, project: ProjectState) -> None:
        super().__init__()
        self.settings, self.project = settings, project
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(4)
        root.setContentsMargins(4, 4, 4, 4)

        # ── Pontszerkesztő (külön modul) ──────────────────────────────────
        self.point_editor = PointEditorWidget(self.project)
        self.point_editor.points_changed.connect(self._on_points_changed)

        # ── Egyezések áttekintője (összesen) ─────────────────────────────
        self.match_overview = MatchOverviewCanvas()

        # ── Splitter: mindkét panel fejléces konténerbe csomagolva ───────
        def _labeled_panel(title: str, widget: QWidget,
                           tooltip: str = "") -> QWidget:
            """Fejléccel ellátott konténer a splitter-résekhez."""
            container = QWidget()
            vbox = QVBoxLayout(container)
            vbox.setContentsMargins(0, 0, 0, 0)
            vbox.setSpacing(0)
            hdr = QLabel(f"  {title}")
            hdr.setFixedHeight(22)
            hdr.setStyleSheet(
                "background:#1e2730; color:#7ab; font-size:11px;"
                "font-weight:bold; border-bottom:1px solid #2e3c4a;"
            )
            if tooltip:
                hdr.setToolTip(tooltip)
            vbox.addWidget(hdr)
            vbox.addWidget(widget, stretch=1)
            return container

        editor_panel = _labeled_panel(
            tr("✏️  Pontszerkesztő  –  kézzel elhelyezett morfpontpárok"),
            self.point_editor,
            tr("Bal képre / jobb képre kattintva helyezz el egyező pontpárokat.\n"
               "Ezek vezérlik a morfozást. Jobb klikk: törlés. Ctrl+Z: visszavonás.")
        )
        overview_panel = _labeled_panel(
            tr("🔍  Illesztési áttekintő  –  összes automatikus egyezés"),
            self.match_overview,
            tr("Az automata illesztés (SuperPoint / SIFT / stb.) által talált\n"
               "összes pontpár: zöld = inlier (RANSAC), piros = outlier.")
        )

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(editor_panel)
        splitter.addWidget(overview_panel)
        splitter.setSizes([500, 200])

        root.addWidget(splitter)

    def load_images(self) -> None:
        """Képek betöltése a pontszerkesztőbe és az overview-ba."""
        self.point_editor.load_images()
        if self.project.image_a is not None and self.project.image_b is not None:
            self.match_overview.set_images(
                self.project.image_a, self.project.image_b)

    def refresh_views(self) -> None:
        """Pontlisták és egyezés-overview frissítése (pl. auto-match után)."""
        self.point_editor.refresh_from_project()

        ra = (np.array(self.project.raw_matches_a)
              if self.project.raw_matches_a else np.empty((0, 2)))
        rb = (np.array(self.project.raw_matches_b)
              if self.project.raw_matches_b else np.empty((0, 2)))
        m  = (np.array(self.project.raw_inlier_mask)
              if self.project.raw_inlier_mask else np.empty((0,), dtype=bool))
        self.match_overview.set_matches(ra, rb, m)

    def _on_points_changed(self) -> None:
        """Pont hozzáadás/törlés/mozgatás esetén hívódik meg."""
        # Update the point pair count display in the editor
        num_pairs = len(self.editor.editor.points_a) if hasattr(self, 'editor') and hasattr(self.editor, 'editor') else 0
        
        # Update the parent window's status bar if accessible
        try:
            # Try to access the main window through parent hierarchy
            parent = self.parent()
            while parent and not hasattr(parent, 'statusBar'):
                parent = parent.parent()
            if parent and hasattr(parent, 'statusBar'):
                parent.statusBar().showMessage(
                    f"{num_pairs} " + tr("pontpár"))
        except:
            pass
        
        # Mark cached morph frames as stale (invalidate preview)
        if hasattr(self, 'preview_canvas'):
            self.preview_canvas = None


class PreviewTab(QWidget):
    def __init__(self, settings: AppSettings) -> None:
        super().__init__()
        self.settings = settings
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        self.preview_canvas = AlignmentPreviewCanvas()

        form = QFormLayout()
        self.view_combo = QComboBox()
        self.view_combo.addItems([tr("Overlay"), tr("Célkép"), tr("Warpolt A")])

        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.0, 1.0)
        self.alpha_spin.setSingleStep(0.05)
        self.alpha_spin.setValue(self.settings.overlay_alpha)

        form.addRow(tr("Nézet:"),         self.view_combo)
        form.addRow(tr("Átlátszóság:"),   self.alpha_spin)

        root.addWidget(self.preview_canvas, stretch=1)
        root.addLayout(form)

        # Képek tárolása az alpha live re-blendhez
        self._target_img: Optional[np.ndarray] = None
        self._warped_img: Optional[np.ndarray] = None

        # Szignál összekötések
        self.view_combo.currentTextChanged.connect(self.preview_canvas.set_view_mode)
        self.alpha_spin.valueChanged.connect(self._on_alpha_changed)

    def set_alignment_images(self, target, warped, overlay) -> None:
        self._target_img = target
        self._warped_img = warped
        self.preview_canvas.set_view_mode(self.view_combo.currentText())
        self.preview_canvas.set_images(target, warped, overlay)

    def _on_alpha_changed(self, value: float) -> None:
        """Alpha csúszó változásakor az overlay-t azonnal újrakeveri."""
        if self._target_img is None or self._warped_img is None:
            return
        try:
            overlay = blend_same_size_images(
                self._warped_img, self._target_img, alpha=float(value))
            self.preview_canvas.set_images(self._target_img, self._warped_img, overlay)
        except RuntimeError:
            pass   # cv2 nincs telepítve – csendesen kihagyja


# ════════════════════════════════════════════════════════════════════════════
#  ExportTab  –  morph animáció előnézete és exportja
# ════════════════════════════════════════════════════════════════════════════

class _MorphPreviewCanvas(QWidget):
    """Egyszerű widget: egyetlen BGR képkocka megjelenítése."""
    def __init__(self) -> None:
        super().__init__()
        self._pixmap: Optional[QPixmap] = None
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)

    def set_frame(self, bgr: Optional[np.ndarray]) -> None:
        self._pixmap = bgr_to_qpixmap(bgr) if bgr is not None else None
        self.update()

    def paintEvent(self, _) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0d1117"))
        if self._pixmap:
            scaled = self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = (self.width()  - scaled.width())  // 2
            y = (self.height() - scaled.height()) // 2
            p.drawPixmap(x, y, scaled)
        else:
            p.setPen(QColor("#444"))
            p.setFont(QFont("Arial", 12))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "Nincs elérhető adat.\nFuttass illesztést először!")


class ExportTab(QWidget):
    """
    Morph animáció előnézete, lejátszása és exportja.

    Az animáció csak akkor elérhető, ha a projekt tartalmaz
    homográfia mátrixot (auto-illesztés után).
    """

    def __init__(self) -> None:
        super().__init__()
        self._frames:      List[np.ndarray] = []
        self._cur_idx:     int = 0
        self._timer        = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._build_ui()

    # ── UI felépítés ─────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ── Előnézeti vászon ─────────────────────────────────────────────────
        self._canvas = _MorphPreviewCanvas()
        root.addWidget(self._canvas, stretch=1)

        # ── Csúszka + keretszám kijelző ──────────────────────────────────────
        slider_row = QHBoxLayout()

        lbl_a = QLabel("A")
        lbl_a.setStyleSheet("color:#aaa;font-weight:bold;")
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.setValue(0)
        self._slider.setTickInterval(1)
        self._slider.valueChanged.connect(self._on_slider)
        lbl_b = QLabel("B")
        lbl_b.setStyleSheet("color:#aaa;font-weight:bold;")

        self._lbl_frame = QLabel("–")
        self._lbl_frame.setFixedWidth(60)
        self._lbl_frame.setAlignment(Qt.AlignmentFlag.AlignRight |
                                     Qt.AlignmentFlag.AlignVCenter)
        self._lbl_frame.setStyleSheet("color:#888;font-size:11px;")

        slider_row.addWidget(lbl_a)
        slider_row.addWidget(self._slider, stretch=1)
        slider_row.addWidget(lbl_b)
        slider_row.addSpacing(8)
        slider_row.addWidget(self._lbl_frame)
        root.addLayout(slider_row)

        # ── Lejátszó + Generálás – ugyanazon sorban, mert szorosan összetartoznak ──
        # Workflow: generál → csúszka / lejátszás → exportál
        play_row = QHBoxLayout()
        play_row.setSpacing(6)

        self._btn_prev = QPushButton("◀")
        self._btn_prev.setFixedWidth(36)
        self._btn_prev.setFixedHeight(30)
        self._btn_prev.clicked.connect(lambda: self._step(-1))

        self._btn_next = QPushButton("▶")
        self._btn_next.setFixedWidth(36)
        self._btn_next.setFixedHeight(30)
        self._btn_next.clicked.connect(lambda: self._step(1))

        self._btn_play = QPushButton("▶  Lejátszás")
        self._btn_play.setCheckable(True)
        self._btn_play.toggled.connect(self._on_play_toggle)
        self._btn_play.setFixedHeight(30)

        play_row.addWidget(self._btn_prev)
        play_row.addWidget(self._btn_next)
        play_row.addSpacing(8)
        play_row.addWidget(self._btn_play)

        # Elválasztó a lejátszó és a generálás között
        play_sep = QFrame()
        play_sep.setFrameShape(QFrame.Shape.VLine)
        play_sep.setStyleSheet("color:#3a4a5a;")
        play_row.addSpacing(8)
        play_row.addWidget(play_sep)
        play_row.addSpacing(8)

        # Generálás gomb + stale warning – a lejátszó mellé kerül
        self._btn_generate = QPushButton("⚙  Képkockák generálása")
        self._btn_generate.setToolTip(
            "Előnézeti képkockák előállítása a kiválasztott módszerrel.\n"
            "Ezután a lejátszó és az export gombok aktívvá válnak.")
        self._btn_generate.clicked.connect(self._generate)
        self._btn_generate.setFixedHeight(30)

        self._lbl_stale = QLabel("")
        self._lbl_stale.setStyleSheet(
            "color:#e8a020; font-size:11px; font-weight:bold;")
        self._lbl_stale.setToolTip(
            "A beállítások megváltoztak – generáld újra a képkockákat!")

        play_row.addWidget(self._btn_generate)
        play_row.addSpacing(6)
        play_row.addWidget(self._lbl_stale)
        play_row.addStretch()

        root.addLayout(play_row)

        # ── Elválasztó ───────────────────────────────────────────────────────
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.HLine)
        sep1.setStyleSheet("color:#343b45;")
        root.addWidget(sep1)

        # ── Beállítások ──────────────────────────────────────────────────────
        cfg_row = QHBoxLayout()
        cfg_row.setSpacing(16)

        # Képkockák száma
        grp_frames = QGroupBox(tr("Képkockák"))
        fl = QFormLayout(grp_frames)
        fl.setContentsMargins(8, 12, 8, 8)

        self._spin_frames = QSpinBox()
        self._spin_frames.setRange(4, 300)
        self._spin_frames.setValue(cfg("export.defaults.frame_count", 40))
        self._spin_frames.setToolTip("Hány közbülső képkocka legyen az animációban")
        self._spin_frames.valueChanged.connect(self._mark_stale)

        self._spin_fps = QDoubleSpinBox()
        self._spin_fps.setRange(1.0, 60.0)
        self._spin_fps.setValue(cfg("export.defaults.fps", 25.0))
        self._spin_fps.setSingleStep(1.0)
        self._spin_fps.setToolTip(
            "Export FPS – a kimentett MP4/GIF lejátszási sebessége.\n"
            "Az előnézeti csúszkát és lejátszást NEM befolyásolja.\n"
            "Újragenerálás sem szükséges FPS-változás után.")
        # FPS szándékosan nincs _mark_stale-hez kötve:
        # csak exportidőzítést befolyásol, a képkockák tartalmát nem.

        fl.addRow(tr("Képkockák:"),  self._spin_frames)
        fl.addRow(tr("Export FPS:"), self._spin_fps)
        cfg_row.addWidget(grp_frames)

        # ── Módszer + per-módszer beállítások (QStackedWidget) ────────────────
        grp_method = QGroupBox(tr("Morph módszer"))
        ml = QVBoxLayout(grp_method)
        ml.setContentsMargins(8, 12, 8, 8)
        ml.setSpacing(6)

        self._combo_method = QComboBox()
        self._combo_method.addItems([
            "Delaunay háromszög",
            "TPS spline",
            "Optikai folyam",
            "Homográfia",
        ])
        self._combo_method.setCurrentText(
            cfg("export.defaults.method", "Delaunay háromszög"))
        self._combo_method.setToolTip(
            "Delaunay háromszög: klasszikus face-morph, legjobb minőség (pontpár kell)\n"
            "TPS spline: sima globális deformáció, nincs háromszög-határ (pontpár kell)\n"
            "Optikai folyam: dense flow alapú, organikus mozgás (pontpár nélkül is)\n"
            "Homográfia: perspektíva-interpoláció, gyors (homográfia mátrix kell)"
        )
        ml.addWidget(self._combo_method)

        # Per-módszer panel stack
        self._method_stack = QStackedWidget()

        # Panel 0 – Delaunay
        pan_tri = QWidget()
        fl_tri = QFormLayout(pan_tri)
        fl_tri.setContentsMargins(0, 4, 0, 0)
        self._lbl_tri_info = QLabel("–")
        self._lbl_tri_info.setStyleSheet("color:#7bc; font-size:11px;")
        self._lbl_tri_info.setWordWrap(True)
        fl_tri.addRow("Pontpárok:", self._lbl_tri_info)
        self._method_stack.addWidget(pan_tri)

        # Panel 1 – TPS spline
        pan_tps = QWidget()
        fl_tps  = QFormLayout(pan_tps)
        fl_tps.setContentsMargins(0, 4, 0, 0)

        self._lbl_tps_info = QLabel("–")
        self._lbl_tps_info.setStyleSheet("color:#7bc; font-size:11px;")
        self._lbl_tps_info.setWordWrap(True)
        fl_tps.addRow(tr("Pontpárok:"), self._lbl_tps_info)

        self._spin_tps_smoothing = QDoubleSpinBox()
        self._spin_tps_smoothing.setRange(0.0, 100.0)
        self._spin_tps_smoothing.setSingleStep(0.5)
        self._spin_tps_smoothing.setDecimals(1)
        self._spin_tps_smoothing.setValue(0.0)
        self._spin_tps_smoothing.setToolTip(
            "0.0 = egzakt interpoláció (átmegy minden kontrollponton)\n"
            ">0  = simítás: outlier-pontok hatása csökken, de az illesztés lazább.\n"
            "Ajánlott: 0.0–2.0 az első próbákhoz.")
        fl_tps.addRow(tr("Simítás:"), self._spin_tps_smoothing)

        self._combo_tps_scale = QComboBox()
        self._combo_tps_scale.addItems([tr("Teljes (lassabb)"), tr("Fél (gyorsabb)"), tr("Negyed (leggyorsabb)")])
        self._combo_tps_scale.setCurrentText(tr("Fél (gyorsabb)"))
        self._combo_tps_scale.setToolTip(
            "A remap-térkép számítási felbontása.\n"
            "Fél felbontáson ~4× gyorsabb, alig észrevehető minőségveszteséggel.\n"
            "Nagyon finom részletekhez (textúra, kis pontok) érdemes Teljes-t választani.")
        fl_tps.addRow(tr("Felbontás:"), self._combo_tps_scale)
        self._method_stack.addWidget(pan_tps)

        # Panel 2 – Optikai folyam
        pan_flow = QWidget()
        fl_flow = QFormLayout(pan_flow)
        fl_flow.setContentsMargins(0, 4, 0, 0)
        self._combo_flow_quality = QComboBox()
        self._combo_flow_quality.addItems([tr("Gyors"), tr("Normál"), tr("Részletes")])
        self._combo_flow_quality.setCurrentText(tr("Normál"))
        self._combo_flow_quality.setToolTip(
            "Farneback optikai folyam (cv2.calcOpticalFlowFarneback) előbeállítások.\n\n"
            "Gyors  – winsize=15, levels=3, poly_n=5, σ=1.1, iter=3\n"
            "  Az OpenCV tutorial alapértelmezett értékeit követi.\n\n"
            "Normál – winsize=21, levels=4, poly_n=7, σ=1.5, iter=4\n"
            "  Kiegyensúlyozott minőség/sebesség; poly_n=7 + σ=1.5\n"
            "  a Farneback 2003 cikk poly_n=7 ajánlásának megfelelő.\n\n"
            "Részletes – winsize=33, levels=5, poly_n=7, σ=1.5, iter=5\n"
            "  Maximális minőség; poly_n=7 esetén σ=1.5 az ajánlott\n"
            "  érték (Farneback 2003 és az OpenCV forráskód alapján).\n\n"
            "Forrás: OpenCV optikai folyam tutorial, Gunnar Farneback\n"
            "'Two-Frame Motion Estimation Based on Polynomial Expansion'\n"
            "(SCIA 2003), és LearnOpenCV.com benchmarkok.")
        fl_flow.addRow("Minőség:", self._combo_flow_quality)
        self._method_stack.addWidget(pan_flow)

        # Panel 2 – Homográfia
        pan_hom = QWidget()
        fl_hom  = QFormLayout(pan_hom)
        fl_hom.setContentsMargins(0, 4, 0, 0)
        lbl_hom = QLabel("Homográfia mátrix\nszükséges (auto-illesztés után)")
        lbl_hom.setStyleSheet("color:#7bc; font-size:11px;")
        fl_hom.addRow(lbl_hom)
        self._method_stack.addWidget(pan_hom)

        ml.addWidget(self._method_stack)
        self._combo_method.currentIndexChanged.connect(
            self._method_stack.setCurrentIndex)
        self._combo_method.currentIndexChanged.connect(self._mark_stale)
        self._combo_flow_quality.currentIndexChanged.connect(self._mark_stale)
        self._spin_tps_smoothing.valueChanged.connect(self._mark_stale)
        self._combo_tps_scale.currentIndexChanged.connect(self._mark_stale)
        cfg_row.addWidget(grp_method)

        # ── Easing + ping-pong ────────────────────────────────────────────────
        grp_ease = QGroupBox(tr("Animáció"))
        el = QFormLayout(grp_ease)
        el.setContentsMargins(8, 12, 8, 8)

        self._combo_ease = QComboBox()
        self._combo_ease.addItems(list(EASING_FUNCS.keys()))
        self._combo_ease.setCurrentText(cfg("export.defaults.easing", "S-görbe"))
        self._combo_ease.setToolTip(
            "Az idő→pozíció átmeneti görbe:\n"
            "Lineáris: egyenletes tempó\n"
            "S-görbe: lassú start + lassú vég (ajánlott)")

        self._chk_pingpong = QCheckBox("Ping-pong (A→B→A)")
        self._chk_pingpong.setChecked(cfg("export.defaults.ping_pong", False))
        self._chk_pingpong.setToolTip(
            "Az animáció végén visszafelé is lejátssza (A→B→A hurok)")

        self._combo_ease.currentIndexChanged.connect(self._mark_stale)
        self._chk_pingpong.toggled.connect(self._mark_stale)
        el.addRow("Easing:", self._combo_ease)
        el.addRow("",        self._chk_pingpong)
        cfg_row.addWidget(grp_ease)

        root.addLayout(cfg_row)

        # ── Export gombok ────────────────────────────────────────────────────
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("color:#343b45;")
        root.addWidget(sep2)

        exp_row = QHBoxLayout()
        exp_row.setSpacing(8)

        self._btn_mp4 = QPushButton("🎬  MP4 export…")
        self._btn_mp4.setFixedHeight(34)
        self._btn_mp4.setToolTip("Animáció mentése MP4 videóként (OpenCV)")
        self._btn_mp4.clicked.connect(self._export_mp4)

        self._btn_gif = QPushButton("🎞  GIF export…")
        self._btn_gif.setFixedHeight(34)
        self._btn_gif.setToolTip("Animáció mentése animált GIF-ként (Pillow)")
        self._btn_gif.clicked.connect(self._export_gif)

        self._btn_png = QPushButton("🖼  PNG sorozat…")
        self._btn_png.setFixedHeight(34)
        self._btn_png.setToolTip("Minden képkocka mentése külön PNG fájlként")
        self._btn_png.clicked.connect(self._export_png)

        self._lbl_export_status = QLabel("")
        self._lbl_export_status.setStyleSheet("color:#888;font-size:11px;")

        exp_row.addWidget(self._btn_mp4)
        exp_row.addWidget(self._btn_gif)
        exp_row.addWidget(self._btn_png)
        exp_row.addStretch()
        exp_row.addWidget(self._lbl_export_status)
        root.addLayout(exp_row)

        self._set_export_enabled(False)

    # ── Elavult jelző ────────────────────────────────────────────────────────

    _BTN_GENERATE_NORMAL = (
        "QPushButton{background:#2c3340;color:#eee;padding:4px 10px;"
        "border-radius:4px;font-size:12px;}"
        "QPushButton:hover{background:#3a4252;}"
        "QPushButton:disabled{background:#2e2e2e;color:#555;}"
    )
    _BTN_GENERATE_STALE = (
        "QPushButton{background:#5a3800;color:#ffcc66;padding:4px 10px;"
        "border-radius:4px;font-size:12px;font-weight:bold;"
        "border:1px solid #e8a020;}"
        "QPushButton:hover{background:#7a4d00;}"
        "QPushButton:disabled{background:#2e2e2e;color:#555;}"
    )

    def _mark_stale(self, *_) -> None:
        """Bármely beállítás változott → jelzi hogy újragenerálás kell."""
        if not self._frames:
            # Még nincs adat, nincs mit jelezni
            return
        self._btn_generate.setStyleSheet(self._BTN_GENERATE_STALE)
        self._btn_generate.setText("⚙  Újragenerálás szükséges!")
        self._lbl_stale.setText("⚠  A beállítások megváltoztak – az előnézet elavult.")
        self._lbl_export_status.setText("")

    def _clear_stale(self) -> None:
        """Generálás után visszaállítja a normál állapotot."""
        self._btn_generate.setStyleSheet(self._BTN_GENERATE_NORMAL)
        self._btn_generate.setText("⚙  Képkockák generálása")
        self._lbl_stale.setText("")

    # ── Export-beállítások mentés/visszaállítás ──────────────────────────────

    def get_export_settings(self) -> Dict[str, Any]:
        """Az összes export UI-beállítás dictionary-ként a projektfájlba."""
        return {
            "frame_count": self._spin_frames.value(),
            "fps":         self._spin_fps.value(),
            "easing":      self._combo_ease.currentText(),
            "ping_pong":   self._chk_pingpong.isChecked(),
            "method":      self._combo_method.currentText(),
            "flow_quality":self._combo_flow_quality.currentText(),
        }

    def restore_export_settings(self, d: Dict[str, Any]) -> None:
        """Visszaállítja az export beállításokat betöltött projektadatból."""
        if "frame_count" in d:
            self._spin_frames.setValue(int(d["frame_count"]))
        if "fps" in d:
            self._spin_fps.setValue(float(d["fps"]))
        if "easing" in d:
            idx = self._combo_ease.findText(d["easing"])
            if idx >= 0:
                self._combo_ease.setCurrentIndex(idx)
        if "ping_pong" in d:
            self._chk_pingpong.setChecked(bool(d["ping_pong"]))
        if "method" in d:
            idx = self._combo_method.findText(d["method"])
            if idx >= 0:
                self._combo_method.setCurrentIndex(idx)
        if "flow_quality" in d:
            idx = self._combo_flow_quality.findText(d["flow_quality"])
            if idx >= 0:
                self._combo_flow_quality.setCurrentIndex(idx)

    # ── Adatok fogadása a MainWindow-tól ─────────────────────────────────────

    def set_morph_data(
        self,
        img_a:  np.ndarray,
        img_b:  np.ndarray,
        H:      Optional[np.ndarray]         = None,
        pts_a:  Optional[List[Tuple]]        = None,
        pts_b:  Optional[List[Tuple]]        = None,
    ) -> None:
        """
        Frissíti a forrásadatokat és automatikusan választ módszert.
        Ha vannak pontpárok → Delaunay; ha csak H → Homográfia.
        """
        self._img_a = img_a
        self._img_b = img_b
        self._H     = H
        self._pts_a = pts_a or []
        self._pts_b = pts_b or []

        # Info frissítése a Delaunay és TPS paneleken
        n_pts = len(self._pts_a)
        if n_pts >= 3:
            self._lbl_tri_info.setText(f"{n_pts} pontpár elérhető  ✓")
            self._lbl_tri_info.setStyleSheet("color:#6c6; font-size:11px;")
        else:
            self._lbl_tri_info.setText(f"{n_pts} pontpár  (min. 3 szükséges)")
            self._lbl_tri_info.setStyleSheet("color:#c84; font-size:11px;")
        if n_pts >= 4:
            self._lbl_tps_info.setText(f"{n_pts} pontpár elérhető  ✓")
            self._lbl_tps_info.setStyleSheet("color:#6c6; font-size:11px;")
        else:
            self._lbl_tps_info.setText(f"{n_pts} pontpár  (min. 4 szükséges)")
            self._lbl_tps_info.setStyleSheet("color:#c84; font-size:11px;")

        # Módszer auto-kiválasztás
        if n_pts >= 3:
            self._combo_method.setCurrentText("Delaunay háromszög")
        elif H is not None:
            self._combo_method.setCurrentText("Homográfia")
        else:
            self._combo_method.setCurrentText("Optikai folyam")

        self._generate()

    # ── Belső logika ─────────────────────────────────────────────────────────

    # Farneback paraméter előbeállítások – archmorph_config.toml → [flow.presets.*]
    # Értékek a konfigfájlból olvasódnak; ha nincs konfigfájl, a beépített
    # értékek maradnak (OpenCV tutorial + Farneback 2003 ajánlások alapján).
    _FLOW_PARAMS = {
        tr("Gyors"): dict(
            pyr_scale  = cfg("flow.presets.fast.pyr_scale",  0.5),
            levels     = cfg("flow.presets.fast.levels",     3),
            winsize    = cfg("flow.presets.fast.winsize",    15),
            iterations = cfg("flow.presets.fast.iterations", 3),
            poly_n     = cfg("flow.presets.fast.poly_n",     5),
            poly_sigma = cfg("flow.presets.fast.poly_sigma", 1.1),
        ),
        tr("Normál"): dict(
            pyr_scale  = cfg("flow.presets.normal.pyr_scale",  0.5),
            levels     = cfg("flow.presets.normal.levels",     4),
            winsize    = cfg("flow.presets.normal.winsize",    21),
            iterations = cfg("flow.presets.normal.iterations", 4),
            poly_n     = cfg("flow.presets.normal.poly_n",     7),
            poly_sigma = cfg("flow.presets.normal.poly_sigma", 1.5),
        ),
        tr("Részletes"): dict(
            pyr_scale  = cfg("flow.presets.detailed.pyr_scale",  0.5),
            levels     = cfg("flow.presets.detailed.levels",     5),
            winsize    = cfg("flow.presets.detailed.winsize",    33),
            iterations = cfg("flow.presets.detailed.iterations", 5),
            poly_n     = cfg("flow.presets.detailed.poly_n",     7),
            poly_sigma = cfg("flow.presets.detailed.poly_sigma", 1.5),
        ),
    }

    def _generate(self) -> None:
        if not hasattr(self, "_img_a") or self._img_a is None:
            QMessageBox.information(
                self, tr("Nincs adat"),
                tr("Nincs elérhető illesztési adat.\n"
                "Futtass előbb automatikus illesztést!"))
            return

        n        = self._spin_frames.value()
        easing   = self._combo_ease.currentText()
        pingpong = self._chk_pingpong.isChecked()
        method   = self._combo_method.currentText()

        prog = QProgressDialog(tr("Képkockák generálása…"), None, 0, 0, self)
        prog.setWindowTitle("ArchMorph")
        prog.setWindowModality(Qt.WindowModality.WindowModal)
        prog.setMinimumDuration(300)
        prog.setValue(0)
        QApplication.processEvents()

        try:
            if method == "Delaunay háromszög":
                if len(self._pts_a) < 3:
                    raise ValueError(
                        "Delaunay morphhoz legalább 3 pontpár szükséges!\n"
                        "Adj hozzá anchor pontokat, vagy válassz más módszert.")
                self._frames = generate_morph_frames_triangle(
                    self._img_a, self._img_b,
                    self._pts_a, self._pts_b,
                    n_frames=n, easing=easing, ping_pong=pingpong)
                method_short = "Delaunay"

            elif method == "TPS spline":
                if len(self._pts_a) < 4:
                    raise ValueError(
                        "TPS morphhoz legalább 4 pontpár szükséges!\n"
                        "Adj hozzá anchor pontokat, vagy válassz más módszert.")
                smoothing = self._spin_tps_smoothing.value()
                scale_map = {
                    tr("Teljes (lassabb)"):      1.0,
                    tr("Fél (gyorsabb)"):        0.5,
                    tr("Negyed (leggyorsabb)"): 0.25,
                }
                scale = scale_map.get(self._combo_tps_scale.currentText(), 0.5)
                self._frames = generate_morph_frames_tps(
                    self._img_a, self._img_b,
                    self._pts_a, self._pts_b,
                    n_frames=n, easing=easing, ping_pong=pingpong,
                    smoothing=smoothing, scale=scale)
                method_short = f"TPS (s={smoothing:.1f}, {int(scale*100)}%)"

            elif method == "Optikai folyam":
                quality = self._combo_flow_quality.currentText()
                fp = self._FLOW_PARAMS.get(quality, self._FLOW_PARAMS[tr("Normál")])
                self._frames = generate_morph_frames_flow(
                    self._img_a, self._img_b,
                    n_frames=n, easing=easing, ping_pong=pingpong,
                    **fp)
                method_short = f"Flow ({quality})"

            else:  # Homográfia
                if self._H is None:
                    raise ValueError(
                        "Homográfia mátrix nem elérhető!\n"
                        "Futtass előbb automatikus illesztést.")
                self._frames = generate_morph_frames(
                    self._img_a, self._img_b, self._H,
                    n_frames=n, easing=easing, ping_pong=pingpong)
                method_short = "Homográfia"

            total = len(self._frames)
            self._slider.setRange(0, max(0, total - 1))
            self._cur_idx = 0
            self._slider.setValue(0)
            self._show_frame(0)
            self._set_export_enabled(True)
            self._clear_stale()
            self._lbl_export_status.setText(
                f"{total} képkocka  |  ~{total / self._spin_fps.value():.1f} s"
                f"  |  {method_short}")

        except Exception as exc:
            QMessageBox.critical(self, tr("Generálási hiba"), str(exc))
        finally:
            prog.close()

    def _show_frame(self, idx: int) -> None:
        if not self._frames:
            return
        idx = max(0, min(idx, len(self._frames) - 1))
        self._cur_idx = idx
        self._canvas.set_frame(self._frames[idx])
        self._lbl_frame.setText(f"{idx + 1} / {len(self._frames)}")

    def _on_slider(self, value: int) -> None:
        self._show_frame(value)

    def _step(self, delta: int) -> None:
        if not self._frames:
            return
        new_idx = (self._cur_idx + delta) % len(self._frames)
        self._slider.setValue(new_idx)

    def _on_play_toggle(self, checked: bool) -> None:
        if checked:
            fps = max(1.0, self._spin_fps.value())
            self._timer.start(int(1000.0 / fps))
            self._btn_play.setText("⏹  Stop")
        else:
            self._timer.stop()
            self._btn_play.setText("▶  Lejátszás")

    def _tick(self) -> None:
        if not self._frames:
            return
        next_idx = (self._cur_idx + 1) % len(self._frames)
        self._slider.setValue(next_idx)

    def _set_export_enabled(self, enabled: bool) -> None:
        for btn in (self._btn_mp4, self._btn_gif, self._btn_png,
                    self._btn_play, self._btn_prev, self._btn_next):
            btn.setEnabled(enabled)
        if not enabled:
            self._lbl_export_status.setText("")

    # ── Export műveletek ─────────────────────────────────────────────────────

    def _export_mp4(self) -> None:
        if not self._frames:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, tr("MP4 mentése"), "morph_animation.mp4",
            tr("MP4 videó (*.mp4);;Minden fájl (*)"))
        if not path:
            return
        try:
            export_mp4(self._frames, path, fps=self._spin_fps.value())
            self._lbl_export_status.setText(tr("MP4 elmentve: ") + f"{Path(path).name}")
        except Exception as exc:
            QMessageBox.critical(self, tr("MP4 export hiba"), str(exc))

    def _export_gif(self) -> None:
        if not self._frames:
            return
        gif_fps = min(self._spin_fps.value(), 50.0)  # GIF max ~50 fps
        path, _ = QFileDialog.getSaveFileName(
            self, tr("GIF mentése"), "morph_animation.gif",
            tr("Animált GIF (*.gif);;Minden fájl (*)"))
        if not path:
            return
        try:
            export_gif(self._frames, path, fps=gif_fps)
            self._lbl_export_status.setText(tr("GIF elmentve: ") + f"{Path(path).name}")
        except Exception as exc:
            QMessageBox.critical(self, tr("GIF export hiba"), str(exc))

    def _export_png(self) -> None:
        if not self._frames:
            return
        folder = QFileDialog.getExistingDirectory(
            self, "Mappa kijelölése a PNG sorozatnak")
        if not folder:
            return
        try:
            export_png_sequence(self._frames, folder)
            self._lbl_export_status.setText(
                f"{len(self._frames)} PNG elmentve → {Path(folder).name}/")
        except Exception as exc:
            QMessageBox.critical(self, tr("PNG export hiba"), str(exc))


# ════════════════════════════════════════════════════════════════════════════
#  Projekt-információ dialógus
# ════════════════════════════════════════════════════════════════════════════

class ProjectInfoDialog(QDialog):
    """
    Egyszerű dialóg a projekt nevének és megjegyzésének szerkesztéséhez.
    Megnyitja a jelenlegi értékeket, és OK esetén visszaadja az újakat.
    """

    def __init__(
        self,
        project_name: str = "",
        notes: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("Projekt információk"))
        self.setMinimumWidth(420)

        form = QFormLayout()
        form.setSpacing(10)
        form.setContentsMargins(16, 16, 16, 8)

        self._name_edit = QLineEdit(project_name)
        self._name_edit.setPlaceholderText(tr("pl. Keleti pályaudvar 1910–2024"))
        self._name_edit.setMaxLength(120)
        form.addRow(tr("Projekt neve:"), self._name_edit)

        self._notes_edit = QTextEdit()
        self._notes_edit.setPlainText(notes)
        self._notes_edit.setPlaceholderText(
            tr("Szabad megjegyzés – fotós, dátum, forrás, stb."))
        self._notes_edit.setFixedHeight(96)
        form.addRow(tr("Megjegyzés:"), self._notes_edit)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        root = QVBoxLayout(self)
        root.setSpacing(4)
        root.addLayout(form)
        root.addWidget(btns)

    # ── Eredmény-lekérdezők ──────────────────────────────────────────────────

    def project_name(self) -> str:
        return self._name_edit.text().strip()

    def notes(self) -> str:
        return self._notes_edit.toPlainText().strip()


# ════════════════════════════════════════════════════════════════════════════
#  Dőlés-korrekció dialógus
# ════════════════════════════════════════════════════════════════════════════

class TiltCorrectionDialog(QDialog):
    """
    Dőlés-korrekció párbeszédablak.

    – Automatikusan megbecsüli a kép dőlési szögét Hough-egyenesek alapján.
    – A felhasználó a spinboxban finomhangolhatja a szöget.
    – Az „Alkalmazás" gomb a korrigált képet adja vissza.
    """

    def __init__(
        self,
        image_bgr: np.ndarray,
        side: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr(f"Dőlés-korrekció – {'A' if side == 'a' else 'B'} kép"))
        self.setMinimumSize(700, 500)
        self._original = image_bgr.copy()
        self._corrected: np.ndarray = image_bgr.copy()

        # ── Elrendezés ──────────────────────────────────────────────────────
        main_lay = QVBoxLayout(self)

        # Előnézet
        self._preview_label = QLabel()
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        main_lay.addWidget(self._preview_label, stretch=1)

        # Vezérlők
        ctrl_lay = QHBoxLayout()
        main_lay.addLayout(ctrl_lay)

        ctrl_lay.addWidget(QLabel(tr("Szög (fok):")))
        self._spin = QDoubleSpinBox()
        self._spin.setRange(-45.0, 45.0)
        self._spin.setSingleStep(0.1)
        self._spin.setDecimals(2)
        self._spin.setSuffix("°")
        self._spin.setFixedWidth(100)
        ctrl_lay.addWidget(self._spin)

        btn_detect = QPushButton(tr("🔍 Automatikus detektálás"))
        btn_detect.setToolTip(tr("Hough-egyenesekkel becsüli meg a dőlési szöget"))
        btn_detect.clicked.connect(self._auto_detect)
        ctrl_lay.addWidget(btn_detect)

        btn_reset = QPushButton(tr("↺ Visszaállítás"))
        btn_reset.clicked.connect(self._reset_angle)
        ctrl_lay.addWidget(btn_reset)

        ctrl_lay.addStretch()

        # Alkalmaz / Mégse
        btn_lay = QHBoxLayout()
        main_lay.addLayout(btn_lay)
        btn_lay.addStretch()
        btn_apply = QPushButton(tr("✔ Alkalmazás"))
        btn_apply.setDefault(True)
        btn_apply.clicked.connect(self.accept)
        btn_lay.addWidget(btn_apply)
        btn_cancel = QPushButton(tr("Mégse"))
        btn_cancel.clicked.connect(self.reject)
        btn_lay.addWidget(btn_cancel)

        # Összekötés
        self._spin.valueChanged.connect(self._update_preview)

        # Első detektálás
        self._auto_detect()

    # ── Belső metódusok ─────────────────────────────────────────────────────

    def _auto_detect(self) -> None:
        angle = detect_tilt_angle(self._original)
        self._spin.blockSignals(True)
        self._spin.setValue(round(angle, 2))
        self._spin.blockSignals(False)
        self._update_preview()

    def _reset_angle(self) -> None:
        self._spin.setValue(0.0)

    def _update_preview(self) -> None:
        angle = self._spin.value()
        if abs(angle) < 1e-4:
            rotated = self._original
        else:
            rotated = rotate_image_by_angle(self._original, angle)
        self._corrected = rotated

        # QPixmap előnézet
        h, w = rotated.shape[:2]
        rgb = rotated[..., ::-1].copy()  # BGR → RGB
        qi = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        px = QPixmap.fromImage(qi)
        avail = self._preview_label.size()
        scaled = px.scaled(avail, Qt.AspectRatioMode.KeepAspectRatio,
                           Qt.TransformationMode.SmoothTransformation)
        self._preview_label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._update_preview()

    # ── Nyilvános API ────────────────────────────────────────────────────────

    def corrected_image(self) -> np.ndarray:
        """Az elfogadott (elforgatott) képet adja vissza."""
        return self._corrected


# ════════════════════════════════════════════════════════════════════════════
#  Munkafolyamat-asszisztens oldalsáv
# ════════════════════════════════════════════════════════════════════════════

class WorkflowPanel(QWidget):
    """
    Bal oldali dokkolható panel – minden lépéshez egy kattintható gomb.

    Állapot szerint:
      • elvégzendő  → narancssárga keret, élénk szöveg  (következő = citromsárga)
      • opcionális  → halvány kék keret, szürke szöveg
      • kész        → nincs keret, zöld pipa, áthúzott szöveg
    """

    # (azonosító, felirat, tooltip)
    STEPS: List[Tuple[str, str, str]] = [
        ("load_a",
         "① Kép A betöltése",
         "Az animáció kiindulóképe (pl. régebbi állapot).\n"
         "Formátum: JPG, PNG, BMP, TIF.\n"
         "A kép mérete bármi lehet – a program skálázza."),

        ("load_b",
         "② Kép B betöltése",
         "Az animáció célképe (pl. mai állapot).\n"
         "Ideális, ha hasonló szögből és felbontásból készült\n"
         "mint az A kép – így pontosabb lesz az illesztés."),

        ("tilt",
         "③ Dőlés-korrekció  ✦ opcionális",
         "Ha valamelyik kép ferén van felvéve (dőlt horizontvonal,\n"
         "ferde épületélek), ez a lépés Hough-transzformációval\n"
         "megbecsüli a szöget és elforgatja a képet.\n\n"
         "Elvégzése JAVÍTJA a GCP igazítás és az automata\n"
         "illesztés pontosságát. A szög kézzel is finomhangolható.\n\n"
         "⚠ GCP igazítás ELŐTT végezd el!"),

        ("gcp",
         "④ GCP igazítás",
         "Ground Control Points – geometriai előigazítás.\n\n"
         "Jelölj meg 4–15 valódi egyező pontot (épületsarkok,\n"
         "ablakkeretek, utcajelzések stb.) a két képen.\n"
         "A program homográfiával warpálja az A képet\n"
         "a B perspektívájába, majd levágja a felesleges széleket.\n\n"
         "✔ ERŐSEN AJÁNLOTT – nélküle az automata illesztés\n"
         "  rosszabb eredményt ad, főleg eltérő perspektívánál."),

        ("edit",
         "⑤ Kézi szerkesztés  ✦ opcionális",
         "A Pontszerkesztő fülön kézzel helyezhetsz el morfpontokat.\n\n"
         "• Jelölj meg épületsarkokat, ablakokat, jellegzetes részleteket\n"
         "• Íveket és vonalláncokat is rajzolhatsz (pl. párkány, tető)\n"
         "• Ezek lesznek az 'anchor' pontok – rögzítik a morfoló hálót\n\n"
         "✔ AJÁNLOTT a GCP után, AZ AUTOMATA ELŐTT –\n"
         "  az automata illesztés sokkal jobb eredményt ad,\n"
         "  ha már vannak kézi anchor pontok referenciának."),

        ("auto_match",
         "⑥ Automata illesztés",
         "Mélytanulás-alapú (SuperPoint/DISK/LoFTR) vagy\n"
         "klasszikus (SIFT) kulcspont-detektor + matcher.\n\n"
         "A program automatikusan megkeresi az összes egyező\n"
         "képrészletet és RANSAC-szűréssel megtartja a legjobb\n"
         "párokat – a kézi pontok mellé adja hozzá ezeket.\n\n"
         "✔ A kézi anchor pontok után futtatva a legjobb:\n"
         "  sűrűbb és egyenletesebb ponthálót eredményez."),

        ("preview",
         "⑦ Előnézet",
         "Az Előnézet fülön valós idejű átmenet-animációt\n"
         "láthatsz az A→B képek között.\n\n"
         "Az alpha csúszkával a keverési arányt állíthatod,\n"
         "így ellenőrizheted, hogy minden pont jól illeszkedik-e."),

        ("export",
         "⑧ Export",
         "Az Export fülön animált GIF-et vagy MP4-et generálhatsz\n"
         "a morfolt képsorból.\n\n"
         "Beállítható: képkockaszám, sebesség, felbontás,\n"
         "visszafelé lejátszás (ping-pong), vízjel."),
    ]

    next_step_requested = pyqtSignal(str)

    # Stílusok – (border-color, text-color, font-weight, background)
    _STYLE_PENDING  = ("#e07820", "#e8a060", "bold",   "#1e1a14")   # narancssárga keret
    _STYLE_NEXT     = ("#f5d020", "#f5e060", "bold",   "#221f0e")   # citromsárga – KÖVETKEZŐ
    _STYLE_OPTIONAL = ("#2a4060", "#4a6080", "normal", "#161b22")   # halvány kék
    _STYLE_DONE     = ("none",    "#3d7a52", "normal", "#141a18")   # nincs keret, zöld

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(215)
        self.setMaximumWidth(270)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 8, 6, 8)
        lay.setSpacing(5)

        # Fejléc
        title = QLabel(tr("  Munkafolyamat"))
        title.setStyleSheet(
            "font-size:11px; font-weight:bold; color:#7a9abb; "
            "padding:4px 2px; border-bottom:1px solid #2a3340;"
        )
        lay.addWidget(title)
        lay.addSpacing(2)

        # Gomb minden lépéshez
        self._btns: dict[str, QPushButton] = {}
        for step_id, label, tooltip in self.STEPS:
            btn = QPushButton(label)
            btn.setToolTip(tooltip)
            btn.setMinimumHeight(34)
            btn.setCheckable(False)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _=False, sid=step_id: self.next_step_requested.emit(sid))
            lay.addWidget(btn)
            self._btns[step_id] = btn

        lay.addStretch()
        self.refresh({})

    # ── Stílus segéd ────────────────────────────────────────────────────────

    @staticmethod
    def _make_style(border: str, color: str, weight: str, bg: str,
                    done: bool = False) -> str:
        text_deco = "line-through" if done else "none"
        border_css = (
            f"border:2px solid {border}; border-radius:5px;"
            if border != "none"
            else "border:1px solid #2a3340; border-radius:5px;"
        )
        return (
            f"QPushButton {{"
            f"  {border_css}"
            f"  background:{bg}; color:{color};"
            f"  font-weight:{weight}; font-size:11px;"
            f"  text-align:left; padding:4px 8px;"
            f"  text-decoration:{text_deco};"
            f"}}"
            f"QPushButton:hover {{"
            f"  background: #2a3040;"
            f"}}"
            f"QPushButton:pressed {{"
            f"  background: #1a2030;"
            f"}}"
        )

    # ── Állapot frissítése ───────────────────────────────────────────────────

    def refresh(self, state: dict) -> None:
        """
        state: {step_id: True | False | None}
          True  → kész
          False → elvégzendő
          None  → opcionális
        """
        next_found = False
        for step_id, label, _tooltip in self.STEPS:
            btn = self._btns[step_id]
            val = state.get(step_id, False)

            if val is True:
                # Kész – nincs keret, zöld szöveg, áthúzva
                style = self._make_style(*self._STYLE_DONE, done=True)
                btn.setText(f"✅  {label}")
                btn.setEnabled(True)

            elif val is None:
                # Opcionális – halvány kék keret
                style = self._make_style(*self._STYLE_OPTIONAL)
                btn.setText(f"◌  {label}")
                btn.setEnabled(True)

            else:
                # Elvégzendő
                if not next_found:
                    # Ez a KÖVETKEZŐ lépés → citromsárga kiemelés
                    style = self._make_style(*self._STYLE_NEXT)
                    btn.setText(f"▶  {label}")
                    next_found = True
                else:
                    style = self._make_style(*self._STYLE_PENDING)
                    btn.setText(f"○  {label}")
                btn.setEnabled(True)

            btn.setStyleSheet(style)


# ════════════════════════════════════════════════════════════════════════════
#  Főablak
# ════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    _IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

    def __init__(self) -> None:
        super().__init__()
        # Load language preference and initialize UI language
        try:
            lang = cfg("ui.language", "hu")
            set_language(lang)
        except:
            set_language("hu")
        
        self.settings      = AppSettings()
        self.project       = ProjectState()
        self._project_path: Optional[Path] = None   # ismert projektfájl → auto-mentés
        self._project_dirty: bool          = False  # van-e nem mentett változás
        self.setWindowTitle(f"{APP_NAME}  {APP_VERSION}")
        self.resize(1380, 820)
        self.setAcceptDrops(True)
        self._build_ui()
        self._build_toolbar()
        self._build_menu()
        self._build_workflow_dock()
        self._apply_dark_theme()

        # ── Témaerkesztő (pipetta üzemmód) ────────────────────────────────
        _config_dir = Path(__file__).parent
        self._theme_editor = ThemeEditor(
            QApplication.instance(), self,
            self._base_qss, _config_dir)

        # Toolbar gomb összekötése (a gomb már létezik, a ThemeEditor most)
        self._theme_btn.clicked.connect(self._theme_editor.toggle)
        self._theme_btn.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        self._theme_btn.customContextMenuRequested.connect(
            lambda pos: self._theme_editor.show_context_menu(
                self._theme_btn.mapToGlobal(pos)))

        # Ctrl+T shortcut (üzemmódon kívül is működik)
        from PyQt6.QtGui import QShortcut, QKeySequence
        _sc = QShortcut(QKeySequence("Ctrl+T"), self)
        _sc.activated.connect(self._theme_editor.toggle)

        self._update_workflow_state()
        self.statusBar().showMessage(tr("Kész."))

    # ── UI felépítése ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.auto_tab    = AutomaticModeTab(self.settings, self.project)
        self.editor_tab  = AdvancedEditorTab(self.settings, self.project)
        self.preview_tab = PreviewTab(self.settings)
        self.export_tab  = ExportTab()

        self.auto_tab.request_auto_match.connect(self.run_auto_match)
        self.editor_tab.point_editor.roi_search_requested.connect(
            self.run_roi_match)
        self.editor_tab.point_editor.dual_roi_search_requested.connect(
            self.run_dual_roi_match)
        self.editor_tab.point_editor.mask_search_requested.connect(
            self.run_mask_match)
        # kép drag & drop a canvasokon → betöltés
        self.editor_tab.point_editor.image_a_drop_requested.connect(
            lambda p: self._load_image_from_path(p, "A"))
        self.editor_tab.point_editor.image_b_drop_requested.connect(
            lambda p: self._load_image_from_path(p, "B"))

        # Tab-ok a munkafolyamat sorrendjében:
        #   ③ Pont szerkesztő  →  ④ Automata illesztés  →  ⑤ Előnézet  →  ⑥ Export
        self.tabs.addTab(self.editor_tab,  tr("⑤  ✏️  Pontszerkesztő"))
        self.tabs.addTab(self.auto_tab,    tr("⑥  ⚡  Automata illesztés"))
        self.tabs.addTab(self.preview_tab, tr("⑦  🔍  Előnézet"))
        self.tabs.addTab(self.export_tab,  tr("⑧  🎬  Export"))

        if _HAS_CONFIG_EDITOR:
            self.settings_tab = ConfigEditorTab(self)
            self.tabs.addTab(self.settings_tab, tr("⚙  Beállítások"))

        # Fülváltáskor automatikus mentés (csak ha van ismert projektfájl)
        self.tabs.currentChanged.connect(self._autosave)

        # Pontok/vonalak változásakor dirty jelzés
        self.editor_tab.point_editor.points_changed.connect(self._mark_dirty)

    def _build_menu(self) -> None:
        mb = self.menuBar()

        # Fájl menü
        file_menu = mb.addMenu(tr("Fájl"))

        act_new_proj = QAction(tr("Új projekt"), self)
        act_new_proj.setShortcut("Ctrl+N")
        act_new_proj.triggered.connect(self.new_project)

        act_close_proj = QAction(tr("Projekt lezárása"), self)
        act_close_proj.triggered.connect(self.close_project)

        act_proj_info = QAction(tr("Projekt neve és megjegyzés…"), self)
        act_proj_info.setShortcut("Ctrl+I")
        act_proj_info.triggered.connect(self.edit_project_info)

        act_load_a = QAction(tr("Kép A betöltése…"), self)
        act_load_a.setShortcut("Ctrl+1")
        act_load_a.triggered.connect(lambda: self.load_image("A"))

        act_load_b = QAction(tr("Kép B betöltése…"), self)
        act_load_b.setShortcut("Ctrl+2")
        act_load_b.triggered.connect(lambda: self.load_image("B"))

        act_save = QAction(tr("Projekt mentése…"), self)
        act_save.setShortcut("Ctrl+S")
        act_save.triggered.connect(self.save_project)

        act_load = QAction(tr("Projekt megnyitása…"), self)
        act_load.setShortcut("Ctrl+O")
        act_load.triggered.connect(self.load_project)

        act_quit = QAction(tr("Kilépés"), self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)

        file_menu.addAction(act_new_proj)
        file_menu.addAction(act_close_proj)
        file_menu.addAction(act_proj_info)
        file_menu.addSeparator()
        file_menu.addActions([act_load_a, act_load_b])
        file_menu.addSeparator()
        file_menu.addActions([act_save, act_load])
        file_menu.addSeparator()
        file_menu.addAction(act_quit)

        # Szerkesztés menü
        edit_menu = mb.addMenu(tr("Szerkesztés"))

        act_undo = QAction(tr("Visszavonás"), self)
        act_undo.setShortcut("Ctrl+Z")
        act_undo.triggered.connect(self._on_undo)
        edit_menu.addAction(act_undo)
        edit_menu.addSeparator()

        act_clear_pts = QAction(tr("Összes pontpár törlése"), self)
        act_clear_pts.triggered.connect(self.clear_all_points)
        edit_menu.addAction(act_clear_pts)

        # Nyelv / Language menü
        lang_menu = mb.addMenu(tr("Language / Nyelv"))

        act_hu = QAction(tr("Magyar"), self)
        act_hu.triggered.connect(lambda: self._set_ui_language("hu"))
        lang_menu.addAction(act_hu)

        act_en = QAction(tr("English"), self)
        act_en.triggered.connect(lambda: self._set_ui_language("en"))
        lang_menu.addAction(act_en)

    # ── Munkafolyamat-panel (dock) ───────────────────────────────────────────

    def _build_workflow_dock(self) -> None:
        try:
            self._workflow_panel = WorkflowPanel()
            self._workflow_panel.next_step_requested.connect(self._on_workflow_next)

            dock = QDockWidget(tr("Munkafolyamat"), self)
            dock.setObjectName("workflow_dock")
            dock.setWidget(self._workflow_panel)
            dock.setFeatures(
                QDockWidget.DockWidgetFeature.DockWidgetMovable |
                QDockWidget.DockWidgetFeature.DockWidgetFloatable |
                QDockWidget.DockWidgetFeature.DockWidgetClosable
            )
            dock.setAllowedAreas(
                Qt.DockWidgetArea.LeftDockWidgetArea |
                Qt.DockWidgetArea.RightDockWidgetArea
            )
            self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
            self._workflow_dock = dock
        except Exception as exc:
            import traceback
            print(f"[WorkflowDock] Nem sikerült létrehozni: {exc}\n{traceback.format_exc()}",
                  file=sys.stderr, flush=True)

    def _update_workflow_state(self) -> None:
        """Frissíti a munkafolyamat-panel állapotát a projekt aktuális státusza alapján."""
        if not hasattr(self, "_workflow_panel"):
            return
        p = self.project
        has_a = p.image_a is not None
        has_b = p.image_b is not None
        has_pts = len(p.anchor_points_a) > 0

        # Kézi szerkesztés: GCP után válik aktívvá (False = elvégzendő),
        # kész ha van legalább 1 kézzel letett pont.
        # Ha GCP még nem volt, marad opcionális (None).
        edit_state: Optional[bool]
        if not p.gcp_done:
            edit_state = None       # GCP előtt: opcionális (halványan jelenik meg)
        elif has_pts:
            edit_state = True       # Van már pont → kész
        else:
            edit_state = False      # GCP kész, de még nincs pont → elvégzendő

        state = {
            "load_a":     has_a,
            "load_b":     has_b,
            "tilt":       None,           # mindig opcionális
            "gcp":        p.gcp_done,
            "edit":       edit_state,
            "auto_match": p.auto_match_done,
            "preview":    has_pts and p.auto_match_done,
            "export":     has_pts and p.auto_match_done,
        }
        self._workflow_panel.refresh(state)

    def _on_workflow_next(self, step_id: str) -> None:
        """A 'Következő lépés' gombra reagálva navigál/indít."""
        if step_id == "load_a":
            self.load_image("A")
        elif step_id == "load_b":
            self.load_image("B")
        elif step_id == "tilt":
            # Opcionális – ha mégis ide kerülünk, A képet ajánljuk
            self._correct_tilt("a")
        elif step_id == "gcp":
            self._open_gcp_alignment()
        elif step_id == "edit":
            self.tabs.setCurrentIndex(0)   # Pontszerkesztő tab
        elif step_id == "auto_match":
            self.tabs.setCurrentIndex(1)   # Automata illesztés tab
        elif step_id in ("preview", "export"):
            idx = {"preview": 2, "export": 3}.get(step_id, 0)
            self.tabs.setCurrentIndex(idx)

    # ── Drag & drop (főablak) ────────────────────────────────────────────────

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                p = url.toLocalFile().lower()
                if p.endswith(self._IMG_EXTS) or p.endswith(".json"):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dragMoveEvent(self, event) -> None:
        event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        paths = [u.toLocalFile() for u in event.mimeData().urls()]
        imgs   = [p for p in paths if p.lower().endswith(self._IMG_EXTS)]
        jsons  = [p for p in paths if p.lower().endswith(".json")]

        # Projekt JSON – az első érvényes .json-t töltjük be
        if jsons:
            self._load_project_from_path(jsons[0])
            event.acceptProposedAction()
            return

        # Képek: 1 kép → A slotba (ha A üres, különben B-be), 2+ → A és B
        if len(imgs) == 1:
            slot = "A" if self.project.image_a is None else "B"
            self._load_image_from_path(imgs[0], slot)
        elif len(imgs) >= 2:
            self._load_image_from_path(imgs[0], "A")
            self._load_image_from_path(imgs[1], "B")
            if len(imgs) > 2:
                self.statusBar().showMessage(
                    f"Kép A+B betöltve  –  {len(imgs) - 2} fájl figyelmen kívül hagyva."
                )
        event.acceptProposedAction()

    def _set_ui_language(self, lang: str) -> None:
        """Set the UI language and save to config."""
        try:
            set_language(lang)
            # Save language preference to config
            try:
                from archmorph_config_loader import save_config
                save_config("ui.language", lang)
            except:
                pass
            # Show restart message
            QMessageBox.information(
                self, 
                tr("Nyelv módosítva / Language Changed"),
                tr("Az alkalmazás újraindítása szükséges a változások érvényre juttatásához.\n\n") +
                "Application restart required for changes to take effect.")
        except Exception as e:
            QMessageBox.warning(self, tr("Hiba / Error"), f"Language change failed: {e}")

    def _build_toolbar(self) -> None:
        tb = QToolBar(tr("Gyorseszközök"))
        tb.setMovable(False)
        tb.setObjectName("main_toolbar")
        self.addToolBar(tb)

        # ── Csoport-felirat segédfüggvény ────────────────────────────────────
        def _group_label(text: str) -> QLabel:
            lbl = QLabel(f"  {text}  ")
            lbl.setStyleSheet(
                "color:#6a8aaa; font-size:10px; font-weight:bold;"
                "padding:0 2px;"
            )
            return lbl

        # ════════════════════════════════════════════════════════════════════
        # PROJEKT – bal oldal: életciklus + fájlkezelés
        # ════════════════════════════════════════════════════════════════════
        tb.addWidget(_group_label("Projekt"))

        act_new = QAction("🆕  Új", self)
        act_new.setToolTip("Új, üres projekt (Ctrl+N)")
        act_new.setShortcut("Ctrl+N")
        act_new.triggered.connect(self.new_project)

        act_open = QAction("📁  Megnyitás", self)
        act_open.setToolTip("Projekt betöltése (Ctrl+O)")
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self.load_project)

        act_save = QAction("💾  Mentés", self)
        act_save.setToolTip("Projekt mentése (Ctrl+S)")
        act_save.setShortcut("Ctrl+S")
        act_save.triggered.connect(self.save_project)

        act_close_proj = QAction("✖  Lezárás", self)
        act_close_proj.setToolTip("Az aktuális projekt lezárása")
        act_close_proj.triggered.connect(self.close_project)

        act_info = QAction("ℹ  Infó", self)
        act_info.setToolTip("Projekt neve és megjegyzés szerkesztése (Ctrl+I)")
        act_info.setShortcut("Ctrl+I")
        act_info.triggered.connect(self.edit_project_info)

        tb.addAction(act_new)
        tb.addAction(act_open)
        tb.addAction(act_save)
        tb.addAction(act_close_proj)
        tb.addAction(act_info)
        tb.addSeparator()

        # ════════════════════════════════════════════════════════════════════
        # ① KÉPEK BETÖLTÉSE
        # ════════════════════════════════════════════════════════════════════
        tb.addWidget(_group_label("① Képek"))

        act_a = QAction(tr("📂  Kép A"), self)
        act_a.setToolTip(tr("Kép A betöltése (Ctrl+1)\nAz animáció kiindulóképe (pl. előtte állapot)"))
        act_a.setShortcut("Ctrl+1")
        act_a.triggered.connect(lambda: self.load_image("A"))

        act_b = QAction(tr("📂  Kép B"), self)
        act_b.setToolTip(tr("Kép B betöltése (Ctrl+2)\nAz animáció célképe (pl. utána állapot)"))
        act_b.setShortcut("Ctrl+2")
        act_b.triggered.connect(lambda: self.load_image("B"))

        act_both = QAction(tr("📂  A+B"), self)
        act_both.setToolTip(tr("Két képfájl egyszerre kijelölése\nAz első lesz Kép A, a második Kép B"))
        act_both.triggered.connect(self.load_images_both)

        tb.addAction(act_a)
        tb.addAction(act_b)
        tb.addAction(act_both)
        tb.addSeparator()

        # ════════════════════════════════════════════════════════════════════
        # ② GEOMETRIAI IGAZÍTÁS – dőlés (opcionális) + GCP (ajánlott)
        # ════════════════════════════════════════════════════════════════════
        tb.addWidget(_group_label("② Igazítás"))

        act_tilt_a = QAction(tr("↕ Dőlés A"), self)
        act_tilt_a.setToolTip(
            tr("Dőlés-korrekció az A képen (opcionális)\n"
               "GCP igazítás előtt érdemes elvégezni."))
        act_tilt_a.triggered.connect(lambda: self._correct_tilt("a"))

        act_tilt_b = QAction(tr("↕ Dőlés B"), self)
        act_tilt_b.setToolTip(
            tr("Dőlés-korrekció a B képen (opcionális)\n"
               "GCP igazítás előtt érdemes elvégezni."))
        act_tilt_b.triggered.connect(lambda: self._correct_tilt("b"))

        act_gcp = QAction(tr("📍  GCP igazítás"), self)
        act_gcp.setToolTip(
            tr("GCP-alapú geometriai igazítás  ← AJÁNLOTT\n\n"
               "Jelölj meg 4–15 egyező pontot a két képen;\n"
               "a program homográfiával igazítja az A képet a B-hez.\n"
               "Utána az automata illesztés lényegesen pontosabb lesz."))
        act_gcp.triggered.connect(self._open_gcp_alignment)

        tb.addAction(act_tilt_a)
        tb.addAction(act_tilt_b)
        tb.addAction(act_gcp)
        tb.addSeparator()

        # ════════════════════════════════════════════════════════════════════
        # ③–⑧ TOVÁBBI LÉPÉSEK – a füleken
        # ════════════════════════════════════════════════════════════════════
        tb.addWidget(_group_label("③–⑧  → Fülek"))

        # ════════════════════════════════════════════════════════════════════
        # 🎨 TÉMAERKESZTŐ – jobb szélen, mindig elérhető
        # ════════════════════════════════════════════════════════════════════
        tb.addSeparator()

        # QToolButton: bal klikk = pipetta toggle, jobb klikk = séma-menü
        self._theme_btn = QToolButton(self)
        self._theme_btn.setText("🎨")
        self._theme_btn.setToolTip(
            "Témaerkesztő  (Ctrl+T)\n"
            "Bal klikk: pipetta üzemmód be-/kikapcsolása\n"
            "Jobb klikk: színsémák mentése / betöltése / visszaállítás"
        )
        self._theme_btn.setFixedSize(36, 28)
        # A clicked és a contextMenu összekötése az _apply_dark_theme UTÁN
        # történik (akkor már létezik self._theme_editor) – ld. __init__
        tb.addWidget(self._theme_btn)

    # ── Dőlés-korrekció ────────────────────────────────────────────────────

    def _correct_tilt(self, side: str) -> None:
        """
        Dőlés-korrekciós dialógust nyit az A vagy B képhez.

        :param side: "a" vagy "b"
        """
        p = self.project
        if side == "a":
            img = p.image_a
        else:
            img = p.image_b

        if img is None:
            QMessageBox.information(
                self,
                tr("Nincs kép"),
                tr(f"Töltsd be a{'z A' if side == 'a' else ' B'} képet először!"),
            )
            return

        dlg = TiltCorrectionDialog(img, side, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        corrected = dlg.corrected_image()
        angle = dlg._spin.value()

        if side == "a":
            # Eredeti megőrzése (ha még nincs mentve)
            if p.image_a_original is None:
                p.image_a_original = p.image_a.copy()
            p.image_a = corrected
            p.image_a_is_modified = True
        else:
            if p.image_b_original is None:
                p.image_b_original = p.image_b.copy()
            p.image_b = corrected
            p.image_b_is_modified = True

        self.editor_tab.load_images()
        self._update_title()
        self._update_workflow_state()
        self.statusBar().showMessage(
            tr(f"Dőlés-korrekció ({'A' if side == 'a' else 'B'} kép): "
               f"{angle:+.2f}° elforgatva  –  új méret: "
               f"{corrected.shape[1]}×{corrected.shape[0]} px"),
            6000,
        )

    # ── GCP-alapú geometriai igazítás ──────────────────────────────────────

    def _open_gcp_alignment(self) -> None:
        """
        GCP (Ground Control Points) alapú igazítás.

        Lépések:
          1. GCPDialog – felhasználó megjelöl 4–15 párpontot
          2. cv2.findHomography (RANSAC) → H mátrix
          3. warpPerspective: A kép B perspektívájába transzformálva
          4. Régi pontpárok törlése (koordináták érvénytelenek)
          5. RANSAC-inlier GCP-k opcionálisan horgonypontként a morfszerkesztőbe
          6. Overlay és preview frissítése
        """
        if self.project.image_a is None or self.project.image_b is None:
            QMessageBox.warning(
                self, tr("Hiányzó kép"), tr("Tölts be mindkét képet!"))
            return

        if cv2 is None:
            QMessageBox.critical(
                self, tr("Hiba"),
                "Az OpenCV (cv2) nincs telepítve – szükséges a homográfia számításhoz.")
            return

        if not _HAS_GCP_DIALOG:
            QMessageBox.critical(
                self, tr("Hiba"), "gcp_dialog.py nem található.")
            return

        dlg = GCPDialog(self.project.image_a, self.project.image_b, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        pairs = dlg.get_pairs()
        if len(pairs) < 4:
            return

        # ── Numpy tömbök ────────────────────────────────────────────────────
        pts_a = np.array([[p[0], p[1]] for p, _ in pairs], dtype=np.float32)
        pts_b = np.array([[p[0], p[1]] for _, p in pairs], dtype=np.float32)

        # ── Adaptív reproj-küszöb (képmérethez arányítva) ───────────────────
        h_a, w_a = self.project.image_a.shape[:2]
        img_diag  = math.hypot(w_a, h_a)
        reproj_px = max(3.0, min(8.0, img_diag * 0.005))   # ~0.5% képátló, 3–8 px között

        # ── Homográfia RANSAC-kal (magasabb megbízhatóság, több iteráció) ────
        H, mask = cv2.findHomography(
            pts_a, pts_b,
            cv2.RANSAC,
            reproj_px,
            confidence  = 0.999,
            maxIters    = 3000,
        )
        if H is None:
            QMessageBox.warning(
                self, tr("Hiba"),
                "A homográfia számítás nem sikerült.\n"
                "Próbálj több / jobban elosztott pontpárt megjelölni,\n"
                "vagy ellenőrizd, hogy valóban megfelelő pontokat jelöltél.")
            return

        n_inliers    = int(mask.sum()) if mask is not None else len(pairs)
        inlier_ratio = n_inliers / max(1, len(pairs))

        # ── Homográfia szanity-ellenőrzés ─────────────────────────────────────
        # 1. Determináns: tükrözés vagy extrém méretarány-változás
        det = float(np.linalg.det(H[:2, :2]))
        det_suspicious = (det <= 0 or abs(math.log(max(abs(det), 1e-9))) > math.log(20))

        # 2. Sarokpont vetítés: befér-e a B kép közelébe?
        h_b_raw, w_b_raw = self.project.image_b.shape[:2]
        corners_a = np.array(
            [[0, 0], [w_a, 0], [w_a, h_a], [0, h_a]],
            dtype=np.float32).reshape(-1, 1, 2)
        corners_b = cv2.perspectiveTransform(corners_a, H).reshape(-1, 2)
        margin    = max(w_b_raw, h_b_raw) * 2
        corner_suspicious = any(
            cx < -margin or cx > w_b_raw + margin
            or cy < -margin or cy > h_b_raw + margin
            for cx, cy in corners_b
        )

        # 3. Inlier arány
        ratio_suspicious = inlier_ratio < 0.5

        warnings: list[str] = []
        if det_suspicious:
            warnings.append(
                f"• Extrém méretarány-változás vagy tükrözés (det = {det:.3f}).")
        if corner_suspicious:
            warnings.append(
                "• A transzformált kép sarkai nagyrészt a B kép területén kívülre kerülnek.")
        if ratio_suspicious:
            warnings.append(
                f"• Alacsony inlier arány: {n_inliers}/{len(pairs)} = "
                f"{inlier_ratio:.0%} (ajánlott: ≥ 50 %).")

        if warnings:
            details = "\n".join(warnings)
            reply = QMessageBox.warning(
                self,
                tr("Gyanús GCP-illesztés"),
                tr(
                    "A RANSAC-illesztés eredménye valószínűleg helytelen:\n\n"
                ) + details + tr(
                    "\n\nEz általában rossz vagy klaszteres GCP-párókból ered.\n"
                    "Tipp: jelölj pontokat a kép négy sarkában is, és töröld\n"
                    "az egyértelműen hibás párokat.\n\n"
                    "Folytatod az igazítást ezzel az eredménnyel?"
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        # ── A kép warpálása B perspektívájába ────────────────────────────────
        h_b, w_b = self.project.image_b.shape[:2]
        warped_a = cv2.warpPerspective(
            self.project.image_a, H, (w_b, h_b),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        # ── Vágás az átfedési területre ──────────────────────────────────────
        # A warp után warped_a szélein fekete sávok keletkezhetnek;
        # mindkét képet a tényleges átfedési területre vágjuk.
        warped_a, image_b_cropped, crop_x, crop_y = crop_images_to_overlap(
            warped_a, self.project.image_b)

        # ── Undo-buffer mentés ───────────────────────────────────────────────
        self.editor_tab.point_editor._push_undo()

        # ── Eredeti képek megőrzése (visszavonáshoz) ─────────────────────────
        if self.project.image_a_original is None:
            self.project.image_a_original = self.project.image_a.copy()
        if self.project.image_b_original is None:
            self.project.image_b_original = self.project.image_b.copy()

        # ── Projekt frissítése ───────────────────────────────────────────────
        self.project.image_a             = warped_a
        self.project.image_a_is_modified = True   # orig. path != memória → mentéskor PNG-be ír
        self.project.image_b             = image_b_cropped
        self.project.image_b_is_modified = True
        self.project.homography_matrix   = H.tolist()

        # Régi pontpárok törlése (koordináták a warp után érvénytelenek)
        self.project.anchor_points_a  = []
        self.project.anchor_points_b  = []
        self.project.raw_matches_a    = []
        self.project.raw_matches_b    = []
        self.project.raw_inlier_mask  = []

        # ── GCP-k horgonypontként a morfszerkesztőbe (opcionális) ────────────
        # Warp + vágás után az inlier-pontok B-koordinátájából le kell vonni
        # a crop offsetet, hogy a vágott képre mutassanak.
        if mask is not None:
            inlier_b = [
                [pts_b[i][0] - crop_x, pts_b[i][1] - crop_y]
                for i in range(len(pairs))
                if mask[i, 0]
            ]
        else:
            inlier_b = [[pb[0] - crop_x, pb[1] - crop_y] for _, pb in pairs]

        if inlier_b:
            reply = QMessageBox.question(
                self,
                tr("GCP-k a morfszerkesztőbe"),
                (
                    f"{n_inliers} RANSAC-inlier GCP pár.\n\n"
                    "Hozzáadjam ezeket horgonypontokként a morfpontszerkesztőhöz?\n"
                    "(Mindkét oldalon azonos pozíción lesznek → stabilizáló hatás.)"
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply == QMessageBox.StandardButton.Yes:
                for pb in inlier_b:
                    self.project.anchor_points_a.append(pb)
                    self.project.anchor_points_b.append(pb)

        # ── Overlay és preview ───────────────────────────────────────────────
        try:
            overlay = blend_same_size_images(warped_a, image_b_cropped, alpha=0.5)
        except Exception:
            overlay = warped_a.copy()

        self.project.aligned_image_a_to_b    = warped_a
        self.project.aligned_overlay_preview = overlay
        self.preview_tab.set_alignment_images(
            image_b_cropped, warped_a, overlay)

        # ── Workflow flag ────────────────────────────────────────────────────
        self.project.gcp_done = True

        # ── UI frissítése ────────────────────────────────────────────────────
        self.editor_tab.load_images()
        self.editor_tab.refresh_views()
        self._update_workflow_state()

        h_crop, w_crop = warped_a.shape[:2]
        self.statusBar().showMessage(
            f"📍 GCP igazítás kész  –  {len(pairs)} pár, {n_inliers} inlier  |  "
            f"Átfedési terület: {w_crop}×{h_crop} px  "
            f"(vágva {w_b-w_crop}×{h_b-h_crop} px-t)"
        )

    def _apply_dark_theme(self) -> None:
        # Az alap QSS-t eltároljuk, hogy a ThemeEditor mindig ehhez adja
        # hozzá az override-okat, ne veszítse el az eredeti stílust.
        self._base_qss = """
            QMainWindow, QWidget   { background-color:#1a1f24; color:#d7dde5; }
            QTabWidget::pane       { border:1px solid #343b45; }
            QTabBar::tab           { background:#252a33; color:#aaa;
                                     padding:6px 16px; border-radius:2px; }
            QTabBar::tab:selected  { background:#1a1f24; color:#eee; }
            QGroupBox              { border:1px solid #343b45; border-radius:4px;
                                     margin-top:8px; padding-top:4px; }
            QGroupBox::title       { color:#9aa; left:8px; }
            QPushButton            { background:#2c3340; color:#eee;
                                     padding:5px 12px; border-radius:4px; }
            QPushButton:hover      { background:#3a4252; }
            QComboBox, QSpinBox, QDoubleSpinBox
                                   { background:#252a33; color:#eee;
                                     border:1px solid #343b45; padding:3px; }
            QCheckBox              { color:#ccc; }
            QStatusBar             { background:#111418; color:#888; }
            QMenuBar               { background:#111418; color:#ccc; }
            QMenuBar::item:selected{ background:#2c3340; }
            QMenu                  { background:#1e232b; color:#eee;
                                     border:1px solid #343b45; }
            QMenu::item:selected   { background:#2c3340; }
            QSplitter::handle      { background:#343b45; }
            QToolBar#main_toolbar  { background:#111418; border:none;
                                     padding:2px 4px; spacing:2px; }
            QToolBar#main_toolbar QToolButton
                                   { background:#252a33; color:#ddd;
                                     padding:4px 10px; border-radius:4px;
                                     font-size:12px; }
            QToolBar#main_toolbar QToolButton:hover
                                   { background:#3a4252; }
            QToolBar#main_toolbar::separator
                                   { background:#343b45; width:1px;
                                     margin:4px 6px; }
            QDockWidget            { background:#131720; color:#ccc;
                                     titlebar-close-icon: none; }
            QDockWidget::title     { background:#1a1f2c; padding:4px 8px;
                                     border-bottom:1px solid #343b45; }
            QDockWidget#workflow_dock QPushButton
                                   { text-align:left; padding:4px 8px;
                                     font-size:11px; }
        """
        QApplication.instance().setStyleSheet(self._base_qss)

    # ── Kép betöltése ────────────────────────────────────────────────────────

    # ── Backend diszpécselés ─────────────────────────────────────────────────

    def _run_backend(
        self,
        img_a:  np.ndarray,
        img_b:  np.ndarray,
        params: dict,
    ):
        """
        Futtatja a kiválasztott illesztő backendet.
        params: get_match_params() eredménye (tartalmazza a backend nevet is).
        Visszatér: (pts_a, pts_b, device) vagy None ha a backend nem elérhető.
        """
        backend = params.get("backend", "")
        # A 'backend', 'use_ransac', 'reproj_threshold' nem backend-paraméter,
        # ezeket kiszűrjük mielőtt **kw-ként átadjuk a függvényeknek.
        kw = {k: v for k, v in params.items()
              if k not in ("backend", "use_ransac", "reproj_threshold")}

        if backend.startswith("SuperPoint"):
            return run_superpoint_lightglue(img_a, img_b, **kw)
        elif backend.startswith("DISK"):
            return run_disk_lightglue(img_a, img_b, **kw)
        elif backend.startswith("LoFTR"):
            return run_loftr(img_a, img_b, **kw)
        elif backend.startswith("SIFT"):
            return run_sift_opencv(img_a, img_b, **kw)
        else:
            QMessageBox.information(
                self, "Backend nem elérhető",
                f'A "{backend}" backend nem ismert.')
            return None

    # ── ROI illesztés (Ctrl+bekeretezés a pont-editorban) ────────────────────

    def run_roi_match(
        self,
        img_polygon:   list,
        side:          str,
        delete_in_roi: bool,
        backend:       str,
    ) -> None:
        """
        Helyi illesztés a kézzel rajzolt területen belül.

        img_polygon   : sokszög csúcsai képkoordinátában [[x,y], ...]
        side          : "A" / "B" – melyik képen értelmezzük a sokszöget
        delete_in_roi : True → törölje a meglévő párokat amelyek a területen belül vannak
        backend       : illesztési algoritmus neve (felülírja az auto_tab beállítását)
        """
        if self.project.image_a is None or self.project.image_b is None:
            QMessageBox.warning(self, tr("Hiányzó kép"), tr("Tölts be mindkét képet!"))
            return
        if len(img_polygon) < 3:
            return

        # Paraméterek az auto_tab-ból, de a backend-et a menüből kapjuk
        params          = self.auto_tab.get_match_params()
        params["backend"] = backend

        n_verts    = len(img_polygon)
        shape_name = tr("téglalap") if n_verts == 4 else f"{n_verts}-" + tr("szög")
        self.statusBar().showMessage(
            tr("ROI illesztés  [") + f"{backend}]  ({shape_name}, {n_verts} " + tr("csúcs)…"))
        QApplication.processEvents()

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            poly_np = np.array(img_polygon, dtype=np.float32)

            # ── Pont-a-sokszögben segédfüggvény ──────────────────────────────
            def _in_poly(px: float, py: float) -> bool:
                if cv2 is None:
                    # cv2 nélkül: AABB fallback
                    xs = poly_np[:, 0]
                    ys = poly_np[:, 1]
                    return bool(xs.min() <= px <= xs.max()
                                and ys.min() <= py <= ys.max())
                return cv2.pointPolygonTest(poly_np, (float(px), float(py)), False) >= 0

            # ── Meglévő pontpárok törlése a ROI-on belül (ha kérték) ─────────
            to_del: list = []
            if delete_in_roi:
                pa_list = self.project.anchor_points_a
                pb_list = self.project.anchor_points_b
                assert len(pa_list) == len(pb_list), f"Length mismatch: pa_list={len(pa_list)}, pb_list={len(pb_list)}"
                for i, (pa, pb) in enumerate(zip(pa_list, pb_list)):
                    test_pt = pa if side == "A" else pb
                    if _in_poly(test_pt[0], test_pt[1]):
                        to_del.append(i)
                if to_del:
                    self.editor_tab.point_editor._push_undo()
                    for idx in sorted(to_del, reverse=True):
                        if idx < len(pa_list): pa_list.pop(idx)
                        if idx < len(pb_list): pb_list.pop(idx)
                    self.editor_tab.refresh_views()

            # ── Automatikus illesztés futtatása a teljes képen ───────────────
            result = self._run_backend(
                self.project.image_a, self.project.image_b, params)
            if result is None:
                return
            pts_a_all, pts_b_all, device = result

            if len(pts_a_all) == 0:
                self.statusBar().showMessage(tr("ROI illesztés: nem talált egyezést."))
                return

            # ── Sokszög-szűrés ────────────────────────────────────────────────
            check_pts = pts_a_all if side == "A" else pts_b_all
            mask = np.array(
                [_in_poly(float(p[0]), float(p[1])) for p in check_pts],
                dtype=bool,
            )
            pts_a_roi = pts_a_all[mask]
            pts_b_roi = pts_b_all[mask]

            if len(pts_a_roi) == 0:
                self.statusBar().showMessage(
                    tr("ROI illesztés: a területen belül nincs egyezés."))
                return

            # ── RANSAC szűrés (ha be van kapcsolva) ──────────────────────────
            if params["use_ransac"] and len(pts_a_roi) >= 4:
                try:
                    pts_a_roi, pts_b_roi, _, _ = filter_matches_with_ransac(
                        pts_a_roi, pts_b_roi,
                        reproj_threshold=params["reproj_threshold"])
                except Exception:
                    pass  # RANSAC hiba → megtartjuk az összes ROI egyezést

            # ── Meglévő pontpárokhoz hozzáfűzés ─────────────────────────────
            self.editor_tab.point_editor._push_undo()
            assert len(pts_a_roi) == len(pts_b_roi), f"ROI points length mismatch: {len(pts_a_roi)} vs {len(pts_b_roi)}"
            for pa, pb in zip(pts_a_roi.tolist(), pts_b_roi.tolist()):
                self.project.anchor_points_a.append(pa)
                self.project.anchor_points_b.append(pb)

            self.editor_tab.refresh_views()
            n_new = len(pts_a_roi)
            n_all = len(self.project.anchor_points_a)
            del_info = (tr("  |  ") + f"{len(to_del)} " + tr("pár törölve")) if delete_in_roi and to_del else ""
            self.statusBar().showMessage(
                tr("ROI illesztés kész  –  +") + f"{n_new} " + tr("új pár  "
                "(összesen: ") + f"{n_all}){del_info}  |  {device}")

        except Exception as exc:
            QMessageBox.critical(self, tr("ROI illesztési hiba"),
                                 log_exception_text(tr("Hiba:"), exc))
            self.statusBar().showMessage(tr("ROI illesztés sikertelen."))
        finally:
            QApplication.restoreOverrideCursor()

    def _update_title(self) -> None:
        """Frissíti az ablakcímet a projekt neve + betöltött képek nevével."""
        # Projekt neve (ha meg van adva) vagy a fájlnév (ha van mentett út)
        if self.project.project_name:
            proj_label = self.project.project_name
        elif self._project_path:
            proj_label = self._project_path.stem
        else:
            proj_label = ""

        # Képfájlok nevei
        img_parts = []
        if self.project.image_a_path:
            img_parts.append(f"A: {self.project.image_a_path.name}")
        if self.project.image_b_path:
            img_parts.append(f"B: {self.project.image_b_path.name}")

        # Összerakás: "ArchMorph Professional  0.5.0  –  Projektnév  [A: ...  |  B: ...]  *"
        parts = []
        if proj_label:
            parts.append(proj_label)
        if img_parts:
            parts.append("  |  ".join(img_parts))
        suffix    = ("  –  " + "    ".join(parts)) if parts else ""
        dirty_str = "  *" if self._project_dirty else ""
        self.setWindowTitle(f"{APP_NAME}  {APP_VERSION}{suffix}{dirty_str}")

    # ── Dirty jelzés ─────────────────────────────────────────────────────────

    def _mark_dirty(self) -> None:
        """Beállítja a nem-mentett-változás jelzőt és frissíti a címsort."""
        if not self._project_dirty:
            self._project_dirty = True
            self._update_title()

    def _has_any_data(self) -> bool:
        """Igaz, ha a projektben van bármilyen adat (képek, pontok, vonalak)."""
        p = self.project
        return bool(
            p.image_a_path or p.image_b_path
            or p.anchor_points_a or p.polylines
        )

    def _confirm_discard(self) -> bool:
        """
        Ha van nem mentett változás, megkérdezi a felhasználót.
        Visszatér True-val ha folytatható (mentett vagy eldobható),
        False-szal ha Cancel-t nyomott.
        """
        if not (self._project_dirty and self._has_any_data()):
            return True
        answer = QMessageBox.question(
            self,
            tr("Nem mentett változások"),
            tr("A projektnek van nem mentett változása.\nMentsük el most?"),
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if answer == QMessageBox.StandardButton.Save:
            self.save_project()
            return True   # mentés után folytatható (még ha dialóg Cancel-lel zárult is)
        if answer == QMessageBox.StandardButton.Discard:
            return True
        return False      # Cancel

    def _reset_project_state(self) -> None:
        """Belső segéd: project adatainak és UI-állapotának teljes visszaállítása."""
        self.project       = ProjectState()
        self._project_path = None
        self._project_dirty = False

        # Canvasok törlése
        self.editor_tab.point_editor.project = self.project
        self.editor_tab.load_images()
        self.editor_tab.refresh_views()
        self._update_title()
        self._update_workflow_state()
        self.statusBar().showMessage(tr("Üres projekt."))

    # ── Új projekt ────────────────────────────────────────────────────────────

    def new_project(self) -> None:
        """Új, üres projekt létrehozása (az aktuális elvész, ha nem mentett)."""
        if not self._confirm_discard():
            return
        self._reset_project_state()
        self.tabs.setCurrentIndex(0)

    # ── Projekt átnevezése / info szerkesztése ────────────────────────────────

    def edit_project_info(self) -> None:
        """Megnyitja a projekt-információ dialógust (név + megjegyzés)."""
        dlg = ProjectInfoDialog(
            project_name=self.project.project_name,
            notes=self.project.notes,
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self.project.project_name = dlg.project_name()
        self.project.notes        = dlg.notes()
        self._mark_dirty()
        self._update_title()

    # ── Projekt lezárása ──────────────────────────────────────────────────────

    def close_project(self) -> None:
        """
        Az aktuális projekt lezárása.
        Ha van bármilyen adat a projektben, mindig megkérdezi: mentse-e.
        """
        if self._has_any_data():
            answer = QMessageBox.question(
                self,
                tr("Projekt lezárása"),
                tr("Elmentse az aktuális projektet lezárás előtt?"),
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Yes,
            )
            if answer == QMessageBox.StandardButton.Cancel:
                return
            if answer == QMessageBox.StandardButton.Yes:
                self.save_project()
                # Ha a mentés dialógban Cancel-t nyomott (path_str üres maradt),
                # ne zárjuk le a projektet
                if self._project_dirty:
                    return
        self._reset_project_state()

    # ── Undo ─────────────────────────────────────────────────────────────────

    def _on_undo(self) -> None:
        """Szerkesztés → Visszavonás menüponthoz."""
        self.editor_tab.point_editor.undo()

    def _load_image_from_path(self, path: str, slot: str) -> None:
        """Belső segédmetódus: betölt egy képfájlt a megadott slotba (A vagy B)."""
        try:
            img = cv_imread_unicode_safe(Path(path))
        except Exception as exc:
            QMessageBox.critical(self, tr("Hiba a kép betöltésekor"), str(exc))
            return
        if slot == "A":
            self.project.image_a             = img
            self.project.image_a_path        = Path(path)
            self.project.image_a_is_modified = False  # lemezről töltöttük, nem módosított
            # Új kép betöltésekor az eredeti puffert és a workflow-flageket reseteljük
            self.project.image_a_original    = None
            self.project.gcp_done            = False
            self.project.auto_match_done     = False
        else:
            self.project.image_b             = img
            self.project.image_b_path        = Path(path)
            self.project.image_b_is_modified = False
            self.project.image_b_original    = None
            self.project.gcp_done            = False
            self.project.auto_match_done     = False
        self.editor_tab.load_images()
        self._update_title()
        self._update_workflow_state()
        # Ha mindkét kép betöltve → ugrás a pontszerkesztőre
        if self.project.image_a is not None and self.project.image_b is not None:
            self.tabs.setCurrentIndex(0)
        self.statusBar().showMessage(
            tr("Kép ") + f"{slot} " + tr("betöltve: ") + f"{Path(path).name}  "
            f"({img.shape[1]}×{img.shape[0]} px)"
        )

    def load_image(self, slot: str) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, tr("Kép ") + f"{slot} " + tr("megnyitása"), "",
            tr("Képfájlok (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;Minden fájl (*)")
        )
        if path:
            self._load_image_from_path(path, slot)

    def load_images_both(self) -> None:
        """Két képfájl egyszerre kijelölése – az első lesz A, a második B."""
        paths, _ = QFileDialog.getOpenFileNames(
            self, tr("Két kép kijelölése (első = A, második = B)"), "",
            tr("Képfájlok (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;Minden fájl (*)")
        )
        if not paths:
            return
        if len(paths) >= 1:
            self._load_image_from_path(paths[0], "A")
        if len(paths) >= 2:
            self._load_image_from_path(paths[1], "B")
        if len(paths) > 2:
            self.statusBar().showMessage(
                tr("Kép A+B betöltve  –  ") + f"{len(paths) - 2} " + tr("fájl figyelmen kívül hagyva.")
            )

    # ── Automatikus illesztés ────────────────────────────────────────────────

    def run_auto_match(self) -> None:
        if self.project.image_a is None or self.project.image_b is None:
            QMessageBox.warning(self, tr("Hiányzó kép"), tr("Tölts be mindkét képet!"))
            return

        params  = self.auto_tab.get_match_params()
        backend = params["backend"]
        self.statusBar().showMessage(tr("Automatikus illesztés folyamatban  [") + f"{backend}]…")
        QApplication.processEvents()

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            result = self._run_backend(
                self.project.image_a, self.project.image_b, params)
            if result is None:
                self.statusBar().showMessage(tr("Illesztés megszakítva."))
                return
            pts_a, pts_b, device = result
            if pts_a is None:
                self.statusBar().showMessage(tr("Illesztés megszakítva."))
                return
            self.project.raw_matches_a = pts_a.tolist()
            self.project.raw_matches_b = pts_b.tolist()

            if params["use_ransac"]:
                in_a, in_b, mask, H = filter_matches_with_ransac(
                    pts_a, pts_b,
                    reproj_threshold=params["reproj_threshold"])
            else:
                in_a, in_b = pts_a, pts_b
                mask = np.ones(len(pts_a), dtype=bool)
                H    = None

            self.project.raw_inlier_mask  = mask.tolist()
            self.project.anchor_points_a  = in_a.tolist()
            self.project.anchor_points_b  = in_b.tolist()
            self.project.homography_matrix= H.tolist() if H is not None else None

            if H is not None:
                warped  = warp_image_to_reference(
                    self.project.image_a, self.project.image_b, H)
                overlay = blend_same_size_images(
                    warped, self.project.image_b, alpha=0.5)
                self.project.aligned_image_a_to_b   = warped
                self.project.aligned_overlay_preview = overlay
                self.preview_tab.set_alignment_images(
                    self.project.image_b, warped, overlay)
                # Export tab frissítése (képek + homográfia + anchor pontok + vonalak)
                # Vonalak renderelési felosztása (30 px-enként)
                poly_a, poly_b = PointEditorWidget.polylines_to_point_pairs(
                    self.project.polylines, step=30.0)
                all_pts_a = [tuple(p) for p in self.project.anchor_points_a] + \
                            [tuple(p) for p in poly_a]
                all_pts_b = [tuple(p) for p in self.project.anchor_points_b] + \
                            [tuple(p) for p in poly_b]
                self.export_tab.set_morph_data(
                    self.project.image_a,
                    self.project.image_b,
                    H     = H,
                    pts_a = all_pts_a,
                    pts_b = all_pts_b,
                )

            self.editor_tab.refresh_views()
            self.tabs.setCurrentIndex(0)   # Pontszerkesztő – azonnal látszanak az új pontok
            self.project.auto_match_done = True
            self._update_workflow_state()
            n = len(self.project.anchor_points_a)
            self.statusBar().showMessage(
                tr("Illesztés kész  –  ") + f"{n} " + tr("pontpár  |  eszköz: ") + f"{device}")

        except Exception as exc:
            QMessageBox.critical(self, tr("Illesztési hiba"),
                                 log_exception_text(tr("Hiba az illesztés során:"), exc))
            self.statusBar().showMessage(tr("Illesztés sikertelen."))
        finally:
            QApplication.restoreOverrideCursor()

    # ── Maszk-alapú keresés (kivágás + LightGlue a területen belül) ─────────

    def run_mask_match(
        self,
        poly_a_norm: list,   # [(nx, ny), ...] norm. (0–1) koordináták, A képre
        poly_b_norm: list,   # [(nx, ny), ...] norm. (0–1) koordináták, B képre
    ) -> None:
        """
        LightGlue futtatása kizárólag a maszk-sokszögek belsejében.

        Stratégia:
          1. A sokszög bounding box-át kivágjuk mindkét képből.
          2. A kivágott képpáron futtatjuk a LightGlue-t.
          3. Csak a sokszög belsejébe eső pontokat tartjuk meg.
          4. A koordinátákat visszatranszformáljuk az eredeti képtérbe.
          5. Hozzáfűzzük a meglévő pontpárokhoz.

        Ez a megközelítés akkor is talál pontokat ahol az egész képre futtatva
        nem találna, mert a LightGlue a kivágott területre fókuszál (nincs
        verseny a jobb texturájú háttér-régiókkal).
        """
        if self.project.image_a is None or self.project.image_b is None:
            QMessageBox.warning(self, tr("Hiányzó kép"), tr("Tölts be mindkét képet!"))
            return
        if len(poly_a_norm) < 3 or len(poly_b_norm) < 3:
            return

        params  = self.auto_tab.get_match_params()
        backend = params["backend"]
        self.statusBar().showMessage(
            tr("Maszk-keresés  [") + f"{backend}]…")
        QApplication.processEvents()

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            img_a = self.project.image_a   # H×W×3 numpy BGR
            img_b = self.project.image_b
            ha, wa = img_a.shape[:2]
            hb, wb = img_b.shape[:2]

            # ── Pixel-koordinátájú sokszögek ──────────────────────────────────
            poly_a_px = np.array([(nx * wa, ny * ha) for nx, ny in poly_a_norm],
                                 dtype=np.float32)
            poly_b_px = np.array([(nx * wb, ny * hb) for nx, ny in poly_b_norm],
                                 dtype=np.float32)

            # ── Bounding box-ok (clamped) ─────────────────────────────────────
            def _bbox(poly, W, H):
                xs, ys = poly[:, 0], poly[:, 1]
                x0, y0 = int(max(0,     np.floor(xs.min()))),  \
                         int(max(0,     np.floor(ys.min())))
                x1, y1 = int(min(W - 1, np.ceil(xs.max()))),   \
                         int(min(H - 1, np.ceil(ys.max())))
                return x0, y0, x1, y1

            ax0, ay0, ax1, ay1 = _bbox(poly_a_px, wa, ha)
            bx0, by0, bx1, by1 = _bbox(poly_b_px, wb, hb)

            if (ax1 - ax0) < 10 or (ay1 - ay0) < 10 or \
               (bx1 - bx0) < 10 or (by1 - by0) < 10:
                QMessageBox.warning(self, tr("Maszk keresés"),
                                    tr("A maszk területe túl kicsi."))
                return

            # ── Kivágott képek ────────────────────────────────────────────────
            crop_a = img_a[ay0:ay1, ax0:ax1].copy()
            crop_b = img_b[by0:by1, bx0:bx1].copy()

            # ── LightGlue futtatása a crop-okon ──────────────────────────────
            result = self._run_backend(crop_a, crop_b, params)
            if result is None:
                self.statusBar().showMessage(tr("Maszk-keresés megszakítva."))
                return
            pts_ca, pts_cb, _device = result

            if len(pts_ca) == 0:
                self.statusBar().showMessage(
                    tr("Maszk-keresés: a területen belül nincs egyezés."))
                return

            # ── Koordináta-visszatranszformálás (crop → eredeti kép) ─────────
            pts_a_full = pts_ca + np.array([ax0, ay0], dtype=np.float32)
            pts_b_full = pts_cb + np.array([bx0, by0], dtype=np.float32)

            # ── Sokszög-szűrés (csak a maszkon belüli pontok) ────────────────
            def _in_poly(pt, poly_np):
                if cv2 is None:
                    xs, ys = poly_np[:, 0], poly_np[:, 1]
                    return bool(xs.min() <= pt[0] <= xs.max()
                                and ys.min() <= pt[1] <= ys.max())
                return cv2.pointPolygonTest(poly_np, (float(pt[0]), float(pt[1])), False) >= 0

            mask_in = np.array(
                [_in_poly(pa, poly_a_px) and _in_poly(pb, poly_b_px)
                 for pa, pb in zip(pts_a_full, pts_b_full)],
                dtype=bool,
            )
            pts_a_in = pts_a_full[mask_in]
            pts_b_in = pts_b_full[mask_in]

            if len(pts_a_in) == 0:
                self.statusBar().showMessage(
                    tr("Maszk-keresés: a sokszög belsejében nincs egyezés."))
                return

            # ── RANSAC (ha be van kapcsolva, ≥4 pont esetén) ─────────────────
            if params["use_ransac"] and len(pts_a_in) >= 4:
                try:
                    pts_a_in, pts_b_in, _, _ = filter_matches_with_ransac(
                        pts_a_in, pts_b_in,
                        reproj_threshold=params["reproj_threshold"])
                except Exception:
                    pass

            # ── Duplikátum-szűrés (ha már van közelben pont) ─────────────────
            existing_a = np.array(self.project.anchor_points_a, dtype=np.float32) \
                         if self.project.anchor_points_a else None
            _DEDUP_R = 6.0

            def _is_dup(pa):
                if existing_a is None or len(existing_a) == 0:
                    return False
                d = np.linalg.norm(existing_a - pa, axis=1)
                return bool(d.min() < _DEDUP_R)

            self.editor_tab.point_editor._push_undo()
            added = 0
            for pa, pb in zip(pts_a_in.tolist(), pts_b_in.tolist()):
                if not _is_dup(pa):
                    self.project.anchor_points_a.append(pa)
                    self.project.anchor_points_b.append(pb)
                    added += 1
                    # Frissítjük az existing_a tömböt
                    existing_a = np.array(self.project.anchor_points_a,
                                          dtype=np.float32)

            self.editor_tab.refresh_views()
            self.project.auto_match_done = True
            self._update_workflow_state()
            n_all = len(self.project.anchor_points_a)
            self.statusBar().showMessage(
                f"{tr('Maszk-keresés kész')}  –  "
                f"+{added} {tr('új pont')} "
                f"({len(pts_ca)} {tr('találat')}, "
                f"{len(pts_a_in)} {tr('belül')}, "
                f"{added} {tr('duplikátum nélkül')})  |  "
                f"{tr('Összesen')}: {n_all} {tr('pontpár')}"
            )

        except Exception as exc:
            QMessageBox.critical(self, tr("Maszk-keresés hiba"),
                                 log_exception_text(tr("Hiba:"), exc))
            self.statusBar().showMessage(tr("Maszk-keresés sikertelen."))
        finally:
            QApplication.restoreOverrideCursor()

    # ── Kétoldali ROI illesztés ──────────────────────────────────────────────

    def run_dual_roi_match(
        self,
        roi_a,          # QRectF képkoordban
        roi_b,          # QRectF képkoordban
        delete_in_roi:  bool,
        backend:        str,
    ) -> None:
        """
        Illesztés csak a két ROI területén belül.
        Mindkét képet a saját ROI-jára vágjuk, az eredmény koordinátákat
        visszaoffseteljük az eredeti képkoordináta-rendszerbe.
        """
        if self.project.image_a is None or self.project.image_b is None:
            QMessageBox.warning(self, tr("Hiányzó kép"), tr("Tölts be mindkét képet!"))
            return

        img_a = self.project.image_a
        img_b = self.project.image_b

        # ROI vágás – integer koordinátákra konvertálva, határokat klampoljuk
        def _crop(img, roi):
            h, w = img.shape[:2]
            x1 = max(0, int(roi.left()))
            y1 = max(0, int(roi.top()))
            x2 = min(w, int(roi.right()))
            y2 = min(h, int(roi.bottom()))
            if x2 <= x1 or y2 <= y1:
                raise ValueError("A ROI kívül esik a képen vagy üres.")
            return img[y1:y2, x1:x2], x1, y1

        try:
            crop_a, off_ax, off_ay = _crop(img_a, roi_a)
            crop_b, off_bx, off_by = _crop(img_b, roi_b)
        except ValueError as exc:
            QMessageBox.warning(self, tr("ROI hiba"), str(exc))
            return

        params  = self.auto_tab.get_match_params()
        params["backend"] = backend
        self.statusBar().showMessage(
            tr("ROI illesztés folyamatban  [") + f"{backend}]  "
            f"A: {crop_a.shape[1]}×{crop_a.shape[0]}  "
            f"B: {crop_b.shape[1]}×{crop_b.shape[0]} " + tr("px…"))
        QApplication.processEvents()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        try:
            result = self._run_backend(crop_a, crop_b, params)
            if result is None:
                self.statusBar().showMessage(tr("ROI illesztés megszakítva."))
                return
            pts_a_crop, pts_b_crop, device = result
            if pts_a_crop is None or len(pts_a_crop) == 0:
                self.statusBar().showMessage(tr("ROI illesztés: nem találhatók pontpárok."))
                return

            # RANSAC szűrés a vágott koordinátákon
            # (minimum 4 pont kell a homográfia becsléshez; kevesebb esetén
            #  az összes egyezést megtartjuk szűrés nélkül)
            if params.get("use_ransac", True) and len(pts_a_crop) >= 4:
                try:
                    pts_a_f, pts_b_f, _, _ = filter_matches_with_ransac(
                        pts_a_crop, pts_b_crop,
                        reproj_threshold=params.get("reproj_threshold", 4.0))
                    if pts_a_f is None or len(pts_a_f) == 0:
                        # RANSAC nem szűrt ki semmit – maradunk az eredetinél
                        pts_a_f, pts_b_f = pts_a_crop, pts_b_crop
                except Exception:
                    # RANSAC hiba (pl. szinguláris mátrix) → szűrés nélkül
                    pts_a_f, pts_b_f = pts_a_crop, pts_b_crop
            else:
                pts_a_f, pts_b_f = pts_a_crop, pts_b_crop

            # Offset visszaadása az eredeti képkoord-rendszerbe
            pts_a_full = pts_a_f + np.array([[off_ax, off_ay]], dtype=np.float64)
            pts_b_full = pts_b_f + np.array([[off_bx, off_by]], dtype=np.float64)

            self.editor_tab.point_editor._push_undo()

            # delete_in_roi: meglévő párok törlése mindkét ROI-n belül
            if delete_in_roi:
                keep = []
                assert len(self.project.anchor_points_a) == len(self.project.anchor_points_b), \
                    f"Point list length mismatch: {len(self.project.anchor_points_a)} vs {len(self.project.anchor_points_b)}"
                for i, (pa, pb) in enumerate(zip(
                        self.project.anchor_points_a,
                        self.project.anchor_points_b)):
                    in_a = roi_a.contains(pa[0], pa[1])
                    in_b = roi_b.contains(pb[0], pb[1])
                    if not (in_a and in_b):
                        keep.append(i)
                self.project.anchor_points_a = [
                    self.project.anchor_points_a[i] for i in keep]
                self.project.anchor_points_b = [
                    self.project.anchor_points_b[i] for i in keep]

            # Új pontpárok hozzáadása
            self.project.anchor_points_a.extend(pts_a_full.tolist())
            self.project.anchor_points_b.extend(pts_b_full.tolist())

            self.editor_tab.refresh_views()
            n_new = len(pts_a_full)
            n_tot = len(self.project.anchor_points_a)
            self.statusBar().showMessage(
                tr("ROI illesztés kész  –  +") + f"{n_new} " + tr("új pár  |  összesen: ") + f"{n_tot}  "
                f"|  " + tr("eszköz: ") + f"{device}")

        except Exception as exc:
            QMessageBox.critical(self, tr("ROI illesztési hiba"),
                                 log_exception_text(tr("Hiba:"), exc))
            self.statusBar().showMessage(tr("ROI illesztés sikertelen."))
        finally:
            QApplication.restoreOverrideCursor()

    # ── Pont törlése (mind) ──────────────────────────────────────────────────

    def clear_all_points(self) -> None:
        if not (self.project.anchor_points_a or self.project.anchor_points_b):
            return
        reply = QMessageBox.question(
            self, tr("Megerősítés"),
            tr("Biztosan törlöd az összes pontpárt?"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.project.anchor_points_a.clear()
            self.project.anchor_points_b.clear()
            self.editor_tab.refresh_views()
            self.statusBar().showMessage(tr("Összes pontpár törölve."))

    # ── Projekt mentése / betöltése ──────────────────────────────────────────

    def _save_project_to(self, path: Path) -> bool:
        """
        Projekt mentése a megadott útvonalra, fájlválasztó dialóg nélkül.
        Ha a memóriában lévő kép eltér az eredeti fájltól (GCP warp / crop),
        a módosított képet PNG-ként menti a JSON mellé.
        Visszatér: True ha sikeres, False ha hiba.
        """
        import datetime
        try:
            json_stem = path.stem
            json_dir  = path.parent
            now_iso   = datetime.datetime.now().isoformat(timespec="seconds")

            # ── Időbélyegek frissítése ────────────────────────────────────────
            if not self.project.created_at:
                self.project.created_at = now_iso
            self.project.modified_at = now_iso

            img_a_path_str = str(self.project.image_a_path) if self.project.image_a_path else None
            img_b_path_str = str(self.project.image_b_path) if self.project.image_b_path else None

            if self.project.image_a_is_modified and self.project.image_a is not None and cv2 is not None:
                mod_path = json_dir / f"{json_stem}_image_a.png"
                cv2.imwrite(str(mod_path), self.project.image_a)
                img_a_path_str                   = str(mod_path)
                self.project.image_a_path        = mod_path
                self.project.image_a_is_modified = False

            if self.project.image_b_is_modified and self.project.image_b is not None and cv2 is not None:
                mod_path = json_dir / f"{json_stem}_image_b.png"
                cv2.imwrite(str(mod_path), self.project.image_b)
                img_b_path_str                   = str(mod_path)
                self.project.image_b_path        = mod_path
                self.project.image_b_is_modified = False

            data = {
                # ── Verzió ──────────────────────────────────────────────────
                "app_version":   APP_VERSION,

                # ── Meta ────────────────────────────────────────────────────
                "meta": {
                    "project_name": self.project.project_name,
                    "notes":        self.project.notes,
                    "created_at":   self.project.created_at,
                    "modified_at":  self.project.modified_at,
                },

                # ── Képek ───────────────────────────────────────────────────
                "image_a":       img_a_path_str,
                "image_b":       img_b_path_str,

                # ── Manuális morfpontok ──────────────────────────────────────
                "points_a":      self.project.anchor_points_a,
                "points_b":      self.project.anchor_points_b,
                "polylines":     self.project.polylines,

                # ── Homográfia (GCP / auto-illesztésből) ─────────────────────
                "homography":    self.project.homography_matrix,

                # ── Automata illesztés nyers eredményei ──────────────────────
                "raw_matches_a":   self.project.raw_matches_a,
                "raw_matches_b":   self.project.raw_matches_b,
                "raw_inlier_mask": self.project.raw_inlier_mask,

                # ── Automata illesztés: összes UI-beállítás ──────────────────
                "auto_params":   self.auto_tab.get_match_params(),

                # ── Export fül beállításai ───────────────────────────────────
                "export":        self.export_tab.get_export_settings(),

                # ── Munkafolyamat-állapot ────────────────────────────────────
                "workflow": {
                    "gcp_done":        self.project.gcp_done,
                    "auto_match_done": self.project.auto_match_done,
                },
            }
            ensure_utf8_json_dump(data, path)
            # Sikeres mentés → dirty törlése
            self._project_dirty = False
            self._update_title()
            return True
        except Exception as exc:
            QMessageBox.critical(self, tr("Mentési hiba"), str(exc))
            return False

    def save_project(self) -> None:
        """Projekt mentése fájlválasztó dialóggal."""
        path_str, _ = QFileDialog.getSaveFileName(
            self, tr("Projekt mentése"), "", tr("ArchMorph projekt (*.json)"))
        if not path_str:
            return
        path = Path(path_str)
        if self._save_project_to(path):
            self._project_path = path
            self.statusBar().showMessage(tr("Projekt mentve: ") + path.name)

    def _autosave(self) -> None:
        """Néma automatikus mentés fülváltáskor, ha van ismert projektfájl."""
        if self._project_path is None:
            return
        if self._save_project_to(self._project_path):
            self.statusBar().showMessage(
                tr("Auto-mentve: ") + self._project_path.name)

    def _load_project_from_path(self, path: str) -> None:
        """Belső segédmetódus: projektfájl betöltése a megadott elérési útról."""
        try:
            data = load_json_utf8(Path(path))

            # ── Meta ─────────────────────────────────────────────────────────
            meta = data.get("meta", {})
            self.project.project_name = meta.get("project_name", "")
            self.project.notes        = meta.get("notes",        "")
            self.project.created_at   = meta.get("created_at",   "")
            self.project.modified_at  = meta.get("modified_at",  "")

            # ── Munkafolyamat-állapot ─────────────────────────────────────────
            wf = data.get("workflow", {})
            self.project.gcp_done        = bool(wf.get("gcp_done",        False))
            self.project.auto_match_done = bool(wf.get("auto_match_done", False))

            pts_a = data.get("points_a", [])
            pts_b = data.get("points_b", [])
            
            # Validate anchor points before assigning
            try:
                clean_a, clean_b = _validate_anchor_points(pts_a, pts_b)
                self.project.anchor_points_a = list(clean_a)
                self.project.anchor_points_b = list(clean_b)
            except ValueError as ve:
                raise ValueError(f"Invalid point data in project: {ve}")
            
            self.project.homography_matrix = data.get("homography")

            # ── Tárolt vonalak ────────────────────────────────────────────────
            raw_polys = data.get("polylines", [])
            self.project.polylines = [
                {"pts_a": list(pl.get("pts_a", [])), "pts_b": list(pl.get("pts_b", []))}
                for pl in raw_polys
                if isinstance(pl, dict)
            ]

            # ── Automata illesztés nyers eredményei ──────────────────────────
            raw_a = data.get("raw_matches_a", [])
            raw_b = data.get("raw_matches_b", [])
            mask  = data.get("raw_inlier_mask", [])
            self.project.raw_matches_a   = [list(p) for p in raw_a]
            self.project.raw_matches_b   = [list(p) for p in raw_b]
            self.project.raw_inlier_mask = list(mask)

            # ── Export beállítások visszaállítása ────────────────────────────
            export_d = data.get("export", {})
            if export_d:
                self.export_tab.restore_export_settings(export_d)

            # ── Automata illesztés összes beállítása ──────────────────────────
            # Régi formátum: "matcher_backend" (csak backend string)
            # Új formátum:   "auto_params" (teljes params dict)
            auto_p = data.get("auto_params")
            if auto_p:
                try:
                    self.auto_tab.restore_params(auto_p)
                except Exception:
                    pass
            else:
                # Visszafelé kompatibilitás régi projektfájlokkal
                backend = data.get("matcher_backend", "")
                if backend:
                    try:
                        idx = self.auto_tab.matcher_combo.findText(backend)
                        if idx >= 0:
                            self.auto_tab.matcher_combo.setCurrentIndex(idx)
                    except Exception:
                        pass

            for slot, key in [("A", "image_a"), ("B", "image_b")]:
                img_path = data.get(key)
                if img_path and Path(img_path).exists():
                    img = cv_imread_unicode_safe(Path(img_path))
                    if slot == "A":
                        self.project.image_a             = img
                        self.project.image_a_path        = Path(img_path)
                        self.project.image_a_is_modified = False
                    else:
                        self.project.image_b             = img
                        self.project.image_b_path        = Path(img_path)
                        self.project.image_b_is_modified = False

            self._project_path  = Path(path)   # auto-mentés engedélyezése ettől fogva
            self._project_dirty = False        # frissen betöltött → nincs módosítás
            # Ha van pontpár → valószínűleg auto-match is volt (visszafelé kompatibilitás)
            if len(self.project.anchor_points_a) > 0 and not self.project.auto_match_done:
                self.project.auto_match_done = True
            self.editor_tab.load_images()
            self.editor_tab.refresh_views()
            self._update_title()
            self._update_workflow_state()
            self.tabs.setCurrentIndex(0)   # Pontszerkesztő
            self.statusBar().showMessage(
                tr("Projekt betöltve: ") + f"{Path(path).name}  "
                f"({len(self.project.anchor_points_a)} " + tr("pontpár)"))
        except Exception as exc:
            QMessageBox.critical(self, tr("Betöltési hiba"), str(exc))

    def load_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, tr("Projekt megnyitása"), "", tr("ArchMorph projekt (*.json)"))
        if path:
            self._load_project_from_path(path)


# ════════════════════════════════════════════════════════════════════════════
#  Belépési pont
# ════════════════════════════════════════════════════════════════════════════

def main() -> int:
    try:
        app = QApplication(sys.argv)
        app.setApplicationName(APP_NAME)
        win = MainWindow()
        win.show()
        return app.exec()
    except Exception as exc:
        import traceback
        msg = f"INDÍTÁSI HIBA:\n{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
        print(msg, file=sys.stderr, flush=True)
        # Ha a Qt már elérhető, megmutatjuk dialógusban is
        try:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(None, "ArchMorph – Indítási hiba", msg[:1200])
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
