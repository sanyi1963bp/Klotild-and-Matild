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
    QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
    QFormLayout, QFrame, QGroupBox, QLabel, QListWidget, QListWidgetItem,
    QMainWindow, QMenu, QMessageBox, QProgressDialog, QPushButton,
    QScrollArea, QSizePolicy, QSlider, QSplitter, QSpinBox, QStackedWidget,
    QStatusBar, QTabWidget, QTextEdit, QToolBar, QVBoxLayout, QHBoxLayout,
    QWidget,
)

# Saját modulok
from point_editor import PointEditorWidget
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
    from crop_dialog import CropDialog
    _HAS_CROP_DIALOG = True
except ImportError:
    CropDialog = None        # type: ignore[assignment,misc]
    _HAS_CROP_DIALOG = False


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
    image_a_path:           Optional[Path]             = None
    image_b_path:           Optional[Path]             = None
    image_a:                Optional[np.ndarray]       = None
    image_b:                Optional[np.ndarray]       = None
    anchor_points_a:        List[List[float]]          = field(default_factory=list)
    anchor_points_b:        List[List[float]]          = field(default_factory=list)
    raw_matches_a:          List[List[float]]          = field(default_factory=list)
    raw_matches_b:          List[List[float]]          = field(default_factory=list)
    raw_inlier_mask:        List[bool]                 = field(default_factory=list)
    homography_matrix:      Optional[List[List[float]]] = None
    aligned_image_a_to_b:   Optional[np.ndarray]       = None
    aligned_overlay_preview: Optional[np.ndarray]      = None
    exclusion_masks:        Dict[str, Any]             = field(default_factory=dict)


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




def crop_images_to_overlap(
    img_a: np.ndarray,
    img_b: np.ndarray,
    pts_a: List[Tuple[float, float]],
    pts_b: List[Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray,
           List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    A két képet a közös, mindkettőn megjelenő területre vágja.

    Algoritmusa:
      1. Homográfiát számít a pontpárokból (A → B tér).
      2. Az A kép érvényes pixeleit betérképezi B koordináta-rendszerébe.
      3. Megkeresi a B képpel való metszet bounding box-ját.
      4. B-t közvetlenül, A warpolt verzióját ugyanerre vágja.
      5. A pontkoordinátákat az új képméretre frissíti.

    Visszatér:
      (img_a_out, img_b_out, new_pts_a, new_pts_b)
      – mindkét kép azonos pixelméretű lesz
      – csak azok a pontpárok maradnak, amelyek a vágott képen belül esnek
    """
    if cv2 is None:
        raise RuntimeError("Az OpenCV (cv2) nincs telepítve.")
    if len(pts_a) < 4:
        raise ValueError(
            "A közös terület meghatározásához legalább 4 pontpár szükséges.\n"
            "Adj hozzá több pontpárt, majd próbáld újra!")

    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]

    # ── 1. Homográfia: A → B tér ──────────────────────────────────────────────
    src = np.float32(pts_a).reshape(-1, 1, 2)
    dst = np.float32(pts_b).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        raise RuntimeError(
            "Nem sikerült homográfiát számítani a pontpárokból.\n"
            "Ellenőrizd, hogy a pontpárok valóban megfelelő helyeken vannak!")

    # ── 2. A érvényes maszkjának vetítése B-be ────────────────────────────────
    mask_a = np.ones((h_a, w_a), dtype=np.uint8) * 255
    warped_mask = cv2.warpPerspective(mask_a, H, (w_b, h_b),
                                      flags=cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=0)

    # ── 3. Közös terület bounding box-a B terében ─────────────────────────────
    ys, xs = np.where(warped_mask > 128)
    if len(xs) == 0:
        raise RuntimeError(
            "A homográfia alapján a két képnek nincs átfedő területe.\n"
            "Ellenőrizd a pontpárokat!")

    x1 = max(0,   int(xs.min()))
    x2 = min(w_b, int(xs.max()) + 1)
    y1 = max(0,   int(ys.min()))
    y2 = min(h_b, int(ys.max()) + 1)

    if (x2 - x1) < 10 or (y2 - y1) < 10:
        raise RuntimeError("Az átfedő terület túl kicsi a vágáshoz.")

    # ── 4. A warpolt változata, majd vágás ────────────────────────────────────
    warped_a  = cv2.warpPerspective(img_a, H, (w_b, h_b),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(0, 0, 0))
    img_a_out = warped_a[y1:y2, x1:x2].copy()
    img_b_out = img_b   [y1:y2, x1:x2].copy()

    # ── 5. Pontkoordináták frissítése ─────────────────────────────────────────
    # pts_b: egyszerű eltolás (x-x1, y-y1)
    # pts_a: H-val B-be transzformálva, majd ugyanaz az eltolás

    pts_a_np = np.float32(pts_a).reshape(-1, 1, 2)
    pts_a_in_b = cv2.perspectiveTransform(pts_a_np, H).reshape(-1, 2)

    new_pts_a: List[Tuple[float, float]] = []
    new_pts_b: List[Tuple[float, float]] = []

    crop_w, crop_h = x2 - x1, y2 - y1
    for (ax, ay), (bx, by) in zip(pts_a_in_b, pts_b):
        nax, nay = ax - x1, ay - y1
        nbx, nby = bx - x1, by - y1
        # Csak azok a párok maradnak, amelyek mindkét oldalon a képen belül esnek
        if (0 <= nax < crop_w and 0 <= nay < crop_h and
                0 <= nbx < crop_w and 0 <= nby < crop_h):
            new_pts_a.append((float(nax), float(nay)))
            new_pts_b.append((float(nbx), float(nby)))

    return img_a_out, img_b_out, new_pts_a, new_pts_b


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

        # ── Backend választó ──────────────────────────────────────────────
        backend_box  = QGroupBox(tr("Illesztési algoritmus (backend)"))
        backend_form = QFormLayout(backend_box)

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
        backend_form.addRow("Backend:", self.matcher_combo)
        root.addWidget(backend_box)

        # ── Per-backend paraméter-panelek ─────────────────────────────────
        params_box  = QGroupBox(tr("Backend paraméterek"))
        params_vbox = QVBoxLayout(params_box)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._panel_superpoint())   # index 0
        self._stack.addWidget(self._panel_disk())         # index 1
        self._stack.addWidget(self._panel_loftr())        # index 2
        self._stack.addWidget(self._panel_sift())         # index 3

        params_vbox.addWidget(self._stack)
        root.addWidget(params_box)

        # ── Közös beállítások (RANSAC + CPU) ──────────────────────────────
        common_box  = QGroupBox(tr("Közös beállítások"))
        common_vbox = QVBoxLayout(common_box)
        common_vbox.addWidget(self._panel_common())
        root.addWidget(common_box)

        # ── Indítás gomb ──────────────────────────────────────────────────
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
            "(Az Szerkesztés → Visszavonás menüvel visszaállítható.)")
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

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.point_editor)
        splitter.addWidget(self.match_overview)
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

        # ── Lejátszó gombok ──────────────────────────────────────────────────
        play_row = QHBoxLayout()
        play_row.setSpacing(6)

        self._btn_play = QPushButton("▶  Lejátszás")
        self._btn_play.setCheckable(True)
        self._btn_play.toggled.connect(self._on_play_toggle)
        self._btn_play.setFixedHeight(30)

        self._btn_prev = QPushButton("◀")
        self._btn_prev.setFixedWidth(36)
        self._btn_prev.setFixedHeight(30)
        self._btn_prev.clicked.connect(lambda: self._step(-1))

        self._btn_next = QPushButton("▶")
        self._btn_next.setFixedWidth(36)
        self._btn_next.setFixedHeight(30)
        self._btn_next.clicked.connect(lambda: self._step(1))

        play_row.addWidget(self._btn_prev)
        play_row.addWidget(self._btn_next)
        play_row.addSpacing(8)
        play_row.addWidget(self._btn_play)
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
        self._spin_fps.setToolTip("Lejátszási sebesség (képkocka/másodperc)")
        # FPS szándékosan nincs _mark_stale-hez kötve:
        # az FPS csak lejátszási sebességet / exportidőzítést befolyásol,
        # a képkockák tartalmát nem érinti.

        fl.addRow("Darab:", self._spin_frames)
        fl.addRow("FPS:",   self._spin_fps)
        cfg_row.addWidget(grp_frames)

        # ── Módszer + per-módszer beállítások (QStackedWidget) ────────────────
        grp_method = QGroupBox(tr("Morph módszer"))
        ml = QVBoxLayout(grp_method)
        ml.setContentsMargins(8, 12, 8, 8)
        ml.setSpacing(6)

        self._combo_method = QComboBox()
        self._combo_method.addItems([
            "Delaunay háromszög",
            "Optikai folyam",
            "Homográfia",
        ])
        self._combo_method.setCurrentText(
            cfg("export.defaults.method", "Delaunay háromszög"))
        self._combo_method.setToolTip(
            "Delaunay háromszög: klasszikus face-morph, legjobb minőség (pontpár kell)\n"
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

        # Panel 1 – Optikai folyam
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

        # ── Generálás gomb ────────────────────────────────────────────────────
        grp_gen = QGroupBox(tr("Generálás"))
        gl = QVBoxLayout(grp_gen)
        gl.setContentsMargins(8, 12, 8, 8)
        self._btn_generate = QPushButton("⚙  Képkockák generálása")
        self._btn_generate.setToolTip(
            "Előnézeti képkockák előállítása a kiválasztott módszerrel")
        self._btn_generate.clicked.connect(self._generate)
        self._btn_generate.setFixedHeight(36)

        self._lbl_stale = QLabel("")
        self._lbl_stale.setStyleSheet("color:#e8a020;font-size:11px;")
        self._lbl_stale.setWordWrap(True)

        gl.addWidget(self._btn_generate)
        gl.addWidget(self._lbl_stale)
        gl.addStretch()
        cfg_row.addWidget(grp_gen)

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

        # Info frissítése a Delaunay panelen
        n_pts = len(self._pts_a)
        if n_pts >= 3:
            self._lbl_tri_info.setText(
                f"{n_pts} pontpár elérhető  ✓")
            self._lbl_tri_info.setStyleSheet("color:#6c6; font-size:11px;")
        else:
            self._lbl_tri_info.setText(
                f"{n_pts} pontpár  (min. 3 szükséges)")
            self._lbl_tri_info.setStyleSheet("color:#c84; font-size:11px;")

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
        
        self.settings = AppSettings()
        self.project  = ProjectState()
        self.setWindowTitle(f"{APP_NAME}  {APP_VERSION}")
        self.resize(1280, 820)
        self.setAcceptDrops(True)
        self._build_ui()
        self._build_toolbar()
        self._build_menu()
        self._apply_dark_theme()
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
        # kép drag & drop a canvasokon → betöltés
        self.editor_tab.point_editor.image_a_drop_requested.connect(
            lambda p: self._load_image_from_path(p, "A"))
        self.editor_tab.point_editor.image_b_drop_requested.connect(
            lambda p: self._load_image_from_path(p, "B"))

        self.tabs.addTab(self.auto_tab,    tr("⚡  Automata illesztés"))
        self.tabs.addTab(self.editor_tab,  tr("✏️  Pontszerkesztő"))
        self.tabs.addTab(self.preview_tab, tr("🔍  Előnézet"))
        self.tabs.addTab(self.export_tab,  tr("🎬  Export"))

        if _HAS_CONFIG_EDITOR:
            self.settings_tab = ConfigEditorTab(self)
            self.tabs.addTab(self.settings_tab, tr("⚙  Beállítások"))

    def _build_menu(self) -> None:
        mb = self.menuBar()

        # Fájl menü
        file_menu = mb.addMenu(tr("Fájl"))

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

        act_a = QAction(tr("📂  Kép A"), self)
        act_a.setToolTip(tr("Kép A betöltése (Ctrl+1)"))
        act_a.triggered.connect(lambda: self.load_image("A"))

        act_b = QAction(tr("📂  Kép B"), self)
        act_b.setToolTip(tr("Kép B betöltése (Ctrl+2)"))
        act_b.triggered.connect(lambda: self.load_image("B"))

        act_both = QAction(tr("📂  Mindkét kép…"), self)
        act_both.setToolTip(
            tr("Két képfájl egyszerre kijelölése\n"
            "Az első lesz Kép A, a második Kép B"))
        act_both.triggered.connect(self.load_images_both)

        tb.addAction(act_a)
        tb.addAction(act_b)
        tb.addAction(act_both)
        tb.addSeparator()

        act_save = QAction("💾  Mentés", self)
        act_save.setToolTip("Projekt mentése (Ctrl+S)")
        act_save.triggered.connect(self.save_project)

        act_open = QAction("📁  Projekt", self)
        act_open.setToolTip("Projekt megnyitása (Ctrl+O)")
        act_open.triggered.connect(self.load_project)

        tb.addAction(act_save)
        tb.addAction(act_open)
        tb.addSeparator()

        act_crop = QAction(tr("✂  Képvágás"), self)
        act_crop.setToolTip(
            tr("Interaktív képvágó megnyitása.\n"
               "Rajzolj téglalapot mindkét képre,\n"
               "majd vágja és méretegyezteti őket."))
        act_crop.triggered.connect(self._crop_to_overlap)
        tb.addAction(act_crop)

    def _crop_to_overlap(self) -> None:
        """Interaktív képvágó párbeszédablak megnyitása."""
        if self.project.image_a is None or self.project.image_b is None:
            QMessageBox.warning(self, tr("Hiányzó kép"), tr("Tölts be mindkét képet!"))
            return

        if not _HAS_CROP_DIALOG:
            QMessageBox.critical(self, tr("Hiba"),
                                 "crop_dialog.py nem található.")
            return

        dlg = CropDialog(self.project.image_a, self.project.image_b, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        img_a_out, img_b_out = dlg.get_results()
        if img_a_out is None or img_b_out is None:
            return

        # Undo mentése a vágás előtti állapotról
        self.editor_tab.point_editor._push_undo()

        # Képek cseréje
        self.project.image_a = img_a_out
        self.project.image_b = img_b_out

        # Pontpárok törlése – a vágás után a koordináták már nem érvényesek
        self.project.anchor_points_a = []
        self.project.anchor_points_b = []
        self.project.raw_matches_a   = []
        self.project.raw_matches_b   = []
        self.project.raw_inlier_mask = []

        # UI frissítése
        self.editor_tab.load_images()
        self.editor_tab.refresh_views()

        h, w = img_a_out.shape[:2]
        self.statusBar().showMessage(
            tr("✂  Vágás kész  –  ") + f"{w}×{h} px")

    def _apply_dark_theme(self) -> None:
        self.setStyleSheet("""
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
        """)

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
        """Frissíti az ablakcímet az épp betöltött képek nevével."""
        parts = []
        if self.project.image_a_path:
            parts.append(f"A: {self.project.image_a_path.name}")
        if self.project.image_b_path:
            parts.append(f"B: {self.project.image_b_path.name}")
        suffix = "  –  " + "  |  ".join(parts) if parts else ""
        self.setWindowTitle(f"{APP_NAME}  {APP_VERSION}{suffix}")

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
            self.project.image_a      = img
            self.project.image_a_path = Path(path)
        else:
            self.project.image_b      = img
            self.project.image_b_path = Path(path)
        self.editor_tab.load_images()
        self._update_title()
        self.tabs.setCurrentIndex(1)
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
                # Export tab frissítése (képek + homográfia + anchor pontok)
                self.export_tab.set_morph_data(
                    self.project.image_a,
                    self.project.image_b,
                    H     = H,
                    pts_a = [tuple(p) for p in self.project.anchor_points_a],
                    pts_b = [tuple(p) for p in self.project.anchor_points_b],
                )

            self.editor_tab.refresh_views()
            self.tabs.setCurrentIndex(1)
            n = len(self.project.anchor_points_a)
            self.statusBar().showMessage(
                tr("Illesztés kész  –  ") + f"{n} " + tr("pontpár  |  eszköz: ") + f"{device}")

        except Exception as exc:
            QMessageBox.critical(self, tr("Illesztési hiba"),
                                 log_exception_text(tr("Hiba az illesztés során:"), exc))
            self.statusBar().showMessage(tr("Illesztés sikertelen."))
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

    def save_project(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, tr("Projekt mentése"), "", tr("ArchMorph projekt (*.json)"))
        if not path:
            return
        data = {
            "app_version": APP_VERSION,
            "image_a":     str(self.project.image_a_path) if self.project.image_a_path else None,
            "image_b":     str(self.project.image_b_path) if self.project.image_b_path else None,
            "points_a":    self.project.anchor_points_a,
            "points_b":    self.project.anchor_points_b,
            "homography":  self.project.homography_matrix,
        }
        try:
            ensure_utf8_json_dump(data, Path(path))
            self.statusBar().showMessage(f"Projekt mentve: {Path(path).name}")
        except Exception as exc:
            QMessageBox.critical(self, "Mentési hiba", str(exc))

    def _load_project_from_path(self, path: str) -> None:
        """Belső segédmetódus: projektfájl betöltése a megadott elérési útról."""
        try:
            data = load_json_utf8(Path(path))
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

            for slot, key in [("A", "image_a"), ("B", "image_b")]:
                img_path = data.get(key)
                if img_path and Path(img_path).exists():
                    img = cv_imread_unicode_safe(Path(img_path))
                    if slot == "A":
                        self.project.image_a      = img
                        self.project.image_a_path = Path(img_path)
                    else:
                        self.project.image_b      = img
                        self.project.image_b_path = Path(img_path)

            self.editor_tab.load_images()
            self.editor_tab.refresh_views()
            self._update_title()
            self.tabs.setCurrentIndex(1)
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
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
