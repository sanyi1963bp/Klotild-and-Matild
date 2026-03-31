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

from PyQt6.QtCore import Qt, QSize, QRectF, pyqtSignal
from PyQt6.QtGui import (
    QAction, QColor, QFont, QImage, QPainter, QPen, QPixmap,
)
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QFrame, QGroupBox, QLabel, QListWidget, QListWidgetItem,
    QMainWindow, QMenu, QMessageBox, QPushButton, QScrollArea, QSizePolicy,
    QSplitter, QSpinBox, QStackedWidget, QStatusBar, QTabWidget, QTextEdit,
    QToolBar, QVBoxLayout, QHBoxLayout, QWidget,
)

# Saját modul
from point_editor import PointEditorWidget

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
    alignment_view_mode:     str   = "Overlay"
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
    pretrained:           str   = "outdoor",
    confidence_threshold: float = 0.5,
    **_,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    LoFTR (Detector-Free Local Feature Matching) via kornia.
    Detektor nélkül sűrű egyezéseket ad – különösen jó textúraszegény
    vagy erősen deformált képpárokon (pl. historikus + modern fotó).

    Telepítés: pip install kornia
    pretrained: "outdoor" (épületek, utcák) vagy "indoor" (belső terek)
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
        self._view_mode  = "Overlay"
        self._all_imgs: Dict[str, Any] = {}   # mindhárom mód képe tárolva

    def sizeHint(self) -> QSize:
        return QSize(600, 400)

    def set_images(self, target, warped, overlay) -> None:
        """Mindhárom nézet képét eltárolja; az aktív módot azonnal mutatja."""
        self._all_imgs = {"Célkép": target, "Warpolt A": warped, "Overlay": overlay}
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
        self.sp_max_kp.setValue(2048)
        self._add_row(form, "Max. kulcspontok:", self.sp_max_kp,
            "A SuperPoint detektor által megtartott kulcspontok maximális száma.\n"
            "Több pont → pontosabb illesztés, de lassabb futás és több GPU-memória.\n"
            "Ajánlott: 1024–4096. Tartomány: 128–8192.")

        self.sp_det_thresh = QDoubleSpinBox()
        self.sp_det_thresh.setRange(0.0001, 0.99)
        self.sp_det_thresh.setSingleStep(0.0001)
        self.sp_det_thresh.setDecimals(4)
        self.sp_det_thresh.setValue(0.0005)
        self._add_row(form, "Detekciós küszöb:", self.sp_det_thresh,
            "A SuperPoint detektor érzékenységi küszöbe (detection_threshold).\n"
            "Alacsonyabb érték → több pont detektálódik, beleértve a gyengébbeket is.\n"
            "Magasabb érték → csak az erős, megbízható sarokpontok maradnak meg.\n"
            "Ajánlott: 0.0001–0.005.")

        self.sp_nms_radius = QSpinBox()
        self.sp_nms_radius.setRange(1, 20)
        self.sp_nms_radius.setValue(4)
        self._add_row(form, "NMS sugár (px):", self.sp_nms_radius,
            "Non-Maximum Suppression sugár pixelben (nms_radius).\n"
            "Megakadályozza, hogy két kulcspont egymáshoz túl közel legyen.\n"
            "Nagyobb érték → ritkább, de egyenletesebb eloszlású pontok.\n"
            "Kisebb érték → sűrűbb pontok, de lehetséges átfedés.\n"
            "Ajánlott: 3–6.")

        self.sp_match_thresh = QDoubleSpinBox()
        self.sp_match_thresh.setRange(0.01, 1.0)
        self.sp_match_thresh.setSingleStep(0.01)
        self.sp_match_thresh.setDecimals(3)
        self.sp_match_thresh.setValue(0.10)
        self._add_row(form, "Match küszöb (LG):", self.sp_match_thresh,
            "A LightGlue párosító szűrési küszöbe (filter_threshold).\n"
            "Alacsonyabb → szigorúbb szűrés: kevesebb, de pontosabb egyezés.\n"
            "Magasabb → több egyezés, de több hibás pár is átcsúszik a szűrőn.\n"
            "Ajánlott: 0.05–0.20.")

        self.sp_depth_conf = QDoubleSpinBox()
        self.sp_depth_conf.setRange(0.5, 1.0)
        self.sp_depth_conf.setSingleStep(0.01)
        self.sp_depth_conf.setDecimals(2)
        self.sp_depth_conf.setValue(0.95)
        self._add_row(form, "Mélység-konfidencia:", self.sp_depth_conf,
            "A LightGlue korai leállás küszöbe mélység irányban (depth_confidence).\n"
            "Ha az egyezések elég megbízhatók, a modell korábban megáll → gyorsabb.\n"
            "1.0 = kikapcsolva (a modell végigfut minden rétegen).\n"
            "Ajánlott: 0.90–0.98.")

        self.sp_width_conf = QDoubleSpinBox()
        self.sp_width_conf.setRange(0.5, 1.0)
        self.sp_width_conf.setSingleStep(0.01)
        self.sp_width_conf.setDecimals(2)
        self.sp_width_conf.setValue(0.99)
        self._add_row(form, "Szélesség-konfidencia:", self.sp_width_conf,
            "A LightGlue korai leállás küszöbe szélesség irányban (width_confidence).\n"
            "Ha az összes pont kezelve van elég biztosan, a modell megáll → gyorsabb.\n"
            "1.0 = kikapcsolva.\n"
            "Ajánlott: 0.95–1.0.")

        return w

    def _panel_disk(self) -> QWidget:
        w    = QWidget()
        form = QFormLayout(w)
        form.setContentsMargins(0, 0, 0, 0)

        self.dk_max_kp = QSpinBox()
        self.dk_max_kp.setRange(128, 8192)
        self.dk_max_kp.setValue(2048)
        self._add_row(form, "Max. kulcspontok:", self.dk_max_kp,
            "A DISK detektor által megtartott kulcspontok maximális száma.\n"
            "A DISK ismétlődő mintákon (ablaksorok, csempék, kövezet) jobban\n"
            "teljesít mint a SuperPoint, mert ezekre van betanítva.\n"
            "Ajánlott: 1024–4096.")

        self.dk_match_thresh = QDoubleSpinBox()
        self.dk_match_thresh.setRange(0.01, 1.0)
        self.dk_match_thresh.setSingleStep(0.01)
        self.dk_match_thresh.setDecimals(3)
        self.dk_match_thresh.setValue(0.10)
        self._add_row(form, "Match küszöb (LG):", self.dk_match_thresh,
            "A LightGlue párosító szűrési küszöbe (filter_threshold).\n"
            "Alacsonyabb → szigorúbb szűrés: kevesebb, de pontosabb egyezés.\n"
            "Magasabb → több egyezés, de több hibás pár is átcsúszik a szűrőn.\n"
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
        self.lf_pretrained.addItems(["outdoor", "indoor"])
        self._add_row(form, "Előtanított modell:", self.lf_pretrained,
            "Az előre tanított LoFTR modell típusa.\n\n"
            "'outdoor' (kültéri): épületek, utcák, terek, tájképek, műemlékek.\n"
            "'indoor' (beltéri): szobák, folyosók, termek, belső terek.\n\n"
            "Válaszd a képeid tartalmához leginkább illőt.")

        self.lf_conf_thresh = QDoubleSpinBox()
        self.lf_conf_thresh.setRange(0.0, 1.0)
        self.lf_conf_thresh.setSingleStep(0.05)
        self.lf_conf_thresh.setDecimals(2)
        self.lf_conf_thresh.setValue(0.50)
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
        self.si_max_kp.setValue(2048)
        self._add_row(form, "Max. kulcspontok:", self.si_max_kp,
            "A SIFT által detektált kulcspontok maximális száma (nfeatures).\n"
            "0 = korlátlan (minden detektált pont megmarad).\n"
            "Ajánlott: 1000–3000 a sebesség és pontosság egyensúlyához.")

        self.si_octave_layers = QSpinBox()
        self.si_octave_layers.setRange(1, 10)
        self.si_octave_layers.setValue(3)
        self._add_row(form, "Oktáv-rétegek:", self.si_octave_layers,
            "Az oktávonkénti rétegek száma a Gauss-skálatérben (nOctaveLayers).\n"
            "Több réteg → finomabb méretarány-invariancia, de lassabb futás.\n"
            "Alapértelmezett: 3 (általában nem kell változtatni).")

        self.si_contrast = QDoubleSpinBox()
        self.si_contrast.setRange(0.001, 0.5)
        self.si_contrast.setSingleStep(0.001)
        self.si_contrast.setDecimals(3)
        self.si_contrast.setValue(0.04)
        self._add_row(form, "Kontraszt küszöb:", self.si_contrast,
            "Alacsony kontrasztú kulcspontok szűrési küszöbe (contrastThreshold).\n"
            "Alacsonyabb → több pont, beleértve a gyenge kontrasztú területeket.\n"
            "Magasabb → csak erős kontrasztú, megbízható pontok maradnak.\n"
            "Ajánlott: 0.02–0.08.")

        self.si_edge = QDoubleSpinBox()
        self.si_edge.setRange(1.0, 50.0)
        self.si_edge.setSingleStep(1.0)
        self.si_edge.setDecimals(1)
        self.si_edge.setValue(10.0)
        self._add_row(form, "Él küszöb:", self.si_edge,
            "Az élszűrő küszöbe (edgeThreshold).\n"
            "Az élek mentén lévő instabil kulcspontokat szűri ki.\n"
            "Magasabb → kevesebb él-jellegű pont törlése (több pont marad).\n"
            "Alacsonyabb → szigorúbb élszűrés (kevesebb, de stabilabb pont).\n"
            "Ajánlott: 5–20.")

        self.si_sigma = QDoubleSpinBox()
        self.si_sigma.setRange(0.5, 5.0)
        self.si_sigma.setSingleStep(0.1)
        self.si_sigma.setDecimals(1)
        self.si_sigma.setValue(1.6)
        self._add_row(form, "Gauss sigma:", self.si_sigma,
            "A Gauss-simítás sigma paramétere a skálatér alaplépésénél.\n"
            "Kisebb (1.2–1.4): kis képeken, kevésbé zajos képeken.\n"
            "Nagyobb (1.8–2.5): nagy, zajos képeken, erősebb simítás.\n"
            "Alapértelmezett: 1.6 (Lowe eredeti értéke).")

        self.si_ratio = QDoubleSpinBox()
        self.si_ratio.setRange(0.5, 0.99)
        self.si_ratio.setSingleStep(0.01)
        self.si_ratio.setDecimals(2)
        self.si_ratio.setValue(0.75)
        self._add_row(form, "Lowe arány küszöb:", self.si_ratio,
            "Lowe-féle arányküszöb a hamis egyezések szűrésére.\n"
            "Egy egyezés elfogadott, ha:\n"
            "  legjobb_távolság < arány × második_legjobb_távolság\n"
            "Alacsonyabb → szigorúbb: kevesebb, de pontosabb egyezés.\n"
            "Magasabb → több egyezés, de több hamis pár is.\n"
            "Ajánlott: 0.65–0.80. Lowe eredeti értéke: 0.75.")

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
        self.ransac_reproj.setValue(4.0)
        self._add_row(form, "Reproj. küszöb (px):", self.ransac_reproj,
            "A RANSAC visszavetítési (reprojekciós) küszöb pixelben.\n"
            "Ha egy pontpár visszavetítési hibája nagyobb ennél, outliernek\n"
            "minősül és törlődik az illesztésből.\n"
            "Kisebb → szigorúbb szűrés (kevesebb pont, de pontosabb).\n"
            "Nagyobb → engedékenyebb szűrés (több pont marad).\n"
            "Ajánlott: 2.0–8.0.")

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
        backend_box  = QGroupBox("Illesztési algoritmus (backend)")
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
        params_box  = QGroupBox("Backend paraméterek")
        params_vbox = QVBoxLayout(params_box)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._panel_superpoint())   # index 0
        self._stack.addWidget(self._panel_disk())         # index 1
        self._stack.addWidget(self._panel_loftr())        # index 2
        self._stack.addWidget(self._panel_sift())         # index 3

        params_vbox.addWidget(self._stack)
        root.addWidget(params_box)

        # ── Közös beállítások (RANSAC + CPU) ──────────────────────────────
        common_box  = QGroupBox("Közös beállítások")
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
        # (Opcionális: itt lehetne pl. live preview triggerelni.)
        pass


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
        self.view_combo.addItems(["Overlay", "Célkép", "Warpolt A"])

        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.0, 1.0)
        self.alpha_spin.setSingleStep(0.05)
        self.alpha_spin.setValue(self.settings.overlay_alpha)

        form.addRow("Nézet:",         self.view_combo)
        form.addRow("Átlátszóság:",   self.alpha_spin)

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
#  Főablak
# ════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.settings = AppSettings()
        self.project  = ProjectState()
        self.setWindowTitle(f"{APP_NAME}  {APP_VERSION}")
        self.resize(1280, 820)
        self._build_ui()
        self._build_menu()
        self._apply_dark_theme()
        self.statusBar().showMessage("Kész.")

    # ── UI felépítése ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.auto_tab    = AutomaticModeTab(self.settings, self.project)
        self.editor_tab  = AdvancedEditorTab(self.settings, self.project)
        self.preview_tab = PreviewTab(self.settings)

        self.auto_tab.request_auto_match.connect(self.run_auto_match)
        self.editor_tab.point_editor.roi_search_requested.connect(
            self.run_roi_match)

        self.tabs.addTab(self.auto_tab,    "⚡  Automata illesztés")
        self.tabs.addTab(self.editor_tab,  "✏️  Pontszerkesztő")
        self.tabs.addTab(self.preview_tab, "🔍  Előnézet")

    def _build_menu(self) -> None:
        mb = self.menuBar()

        # Fájl menü
        file_menu = mb.addMenu("Fájl")

        act_load_a = QAction("Kép A betöltése…", self)
        act_load_a.setShortcut("Ctrl+1")
        act_load_a.triggered.connect(lambda: self.load_image("A"))

        act_load_b = QAction("Kép B betöltése…", self)
        act_load_b.setShortcut("Ctrl+2")
        act_load_b.triggered.connect(lambda: self.load_image("B"))

        act_save = QAction("Projekt mentése…", self)
        act_save.setShortcut("Ctrl+S")
        act_save.triggered.connect(self.save_project)

        act_load = QAction("Projekt megnyitása…", self)
        act_load.setShortcut("Ctrl+O")
        act_load.triggered.connect(self.load_project)

        act_quit = QAction("Kilépés", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)

        file_menu.addActions([act_load_a, act_load_b])
        file_menu.addSeparator()
        file_menu.addActions([act_save, act_load])
        file_menu.addSeparator()
        file_menu.addAction(act_quit)

        # Szerkesztés menü
        edit_menu = mb.addMenu("Szerkesztés")

        act_undo = QAction("Visszavonás", self)
        act_undo.setShortcut("Ctrl+Z")
        act_undo.triggered.connect(self._on_undo)
        edit_menu.addAction(act_undo)
        edit_menu.addSeparator()

        act_clear_pts = QAction("Összes pontpár törlése", self)
        act_clear_pts.triggered.connect(self.clear_all_points)
        edit_menu.addAction(act_clear_pts)

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
            QMessageBox.warning(self, "Hiányzó kép", "Tölts be mindkét képet!")
            return
        if len(img_polygon) < 3:
            return

        # Paraméterek az auto_tab-ból, de a backend-et a menüből kapjuk
        params          = self.auto_tab.get_match_params()
        params["backend"] = backend

        n_verts    = len(img_polygon)
        shape_name = "téglalap" if n_verts == 4 else f"{n_verts}-szög"
        self.statusBar().showMessage(
            f"ROI illesztés  [{backend}]  ({shape_name}, {n_verts} csúcs)…")
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
                self.statusBar().showMessage("ROI illesztés: nem talált egyezést.")
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
                    "ROI illesztés: a területen belül nincs egyezés.")
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
            for pa, pb in zip(pts_a_roi.tolist(), pts_b_roi.tolist()):
                self.project.anchor_points_a.append(pa)
                self.project.anchor_points_b.append(pb)

            self.editor_tab.refresh_views()
            n_new = len(pts_a_roi)
            n_all = len(self.project.anchor_points_a)
            del_info = f"  |  {len(to_del)} pár törölve" if delete_in_roi and to_del else ""
            self.statusBar().showMessage(
                f"ROI illesztés kész  –  +{n_new} új pár  "
                f"(összesen: {n_all}){del_info}  |  {device}")

        except Exception as exc:
            QMessageBox.critical(self, "ROI illesztési hiba",
                                 log_exception_text("Hiba:", exc))
            self.statusBar().showMessage("ROI illesztés sikertelen.")
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

    def load_image(self, slot: str) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, f"Kép {slot} megnyitása", "",
            "Képfájlok (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;Minden fájl (*)"
        )
        if not path:
            return
        try:
            img = cv_imread_unicode_safe(Path(path))
        except Exception as exc:
            QMessageBox.critical(self, "Hiba a kép betöltésekor", str(exc))
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
            f"Kép {slot} betöltve: {Path(path).name}  "
            f"({img.shape[1]}×{img.shape[0]} px)"
        )

    # ── Automatikus illesztés ────────────────────────────────────────────────

    def run_auto_match(self) -> None:
        if self.project.image_a is None or self.project.image_b is None:
            QMessageBox.warning(self, "Hiányzó kép", "Tölts be mindkét képet!")
            return

        params  = self.auto_tab.get_match_params()
        backend = params["backend"]
        self.statusBar().showMessage(f"Automatikus illesztés folyamatban  [{backend}]…")
        QApplication.processEvents()

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            result = self._run_backend(
                self.project.image_a, self.project.image_b, params)
            if result is None:
                self.statusBar().showMessage("Illesztés megszakítva.")
                return
            pts_a, pts_b, device = result
            if pts_a is None:
                self.statusBar().showMessage("Illesztés megszakítva.")
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

            self.editor_tab.refresh_views()
            self.tabs.setCurrentIndex(1)
            n = len(self.project.anchor_points_a)
            self.statusBar().showMessage(
                f"Illesztés kész  –  {n} pontpár  |  eszköz: {device}")

        except Exception as exc:
            QMessageBox.critical(self, "Illesztési hiba",
                                 log_exception_text("Hiba az illesztés során:", exc))
            self.statusBar().showMessage("Illesztés sikertelen.")
        finally:
            QApplication.restoreOverrideCursor()

    # ── Pont törlése (mind) ──────────────────────────────────────────────────

    def clear_all_points(self) -> None:
        if not (self.project.anchor_points_a or self.project.anchor_points_b):
            return
        reply = QMessageBox.question(
            self, "Megerősítés",
            "Biztosan törlöd az összes pontpárt?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.project.anchor_points_a.clear()
            self.project.anchor_points_b.clear()
            self.editor_tab.refresh_views()
            self.statusBar().showMessage("Összes pontpár törölve.")

    # ── Projekt mentése / betöltése ──────────────────────────────────────────

    def save_project(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Projekt mentése", "", "ArchMorph projekt (*.json)")
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

    def load_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Projekt megnyitása", "", "ArchMorph projekt (*.json)")
        if not path:
            return
        try:
            data = load_json_utf8(Path(path))
            self.project.anchor_points_a  = data.get("points_a", [])
            self.project.anchor_points_b  = data.get("points_b", [])
            self.project.homography_matrix= data.get("homography")

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
                f"Projekt betöltve: {Path(path).name}  "
                f"({len(self.project.anchor_points_a)} pontpár)")
        except Exception as exc:
            QMessageBox.critical(self, "Betöltési hiba", str(exc))


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
