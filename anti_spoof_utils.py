"""
anti_spoof_utils.py — Liveness detection matching the exact GitHub demo pipeline.

Fixes vs previous version:
  1. Input must be RGB (not BGR) — model was trained on RGB
  2. Logit index 0 = REAL, index 1 = SPOOF  (previous version was reversed)
  3. Scoring uses logit difference, not softmax probability
  4. Bbox is expanded by 1.5x before cropping (matches demo bbox_expansion_factor)

Confirmed model specs:
  Input  : name='input',  shape=(batch, 3, 128, 128), float32, RGB, normalised /255
  Output : name='output', shape=(batch, 2)  →  [real_logit, spoof_logit]
  Decision: real_logit - spoof_logit > threshold  (default threshold=0 i.e. p=0.5)
"""

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError(
        "onnxruntime is required.\n"
        "Install:  pip install onnxruntime\n"
        "With GPU: pip install onnxruntime-gpu"
    )

# ── Constants ──────────────────────────────────────────────────────────────────
INPUT_SIZE          = (128, 128)
INPUT_NAME          = "input"
OUTPUT_NAME         = "output"
REAL_IDX            = 0          # output[0] = real/live logit
SPOOF_IDX           = 1          # output[1] = spoof logit
DEFAULT_THRESHOLD   = 0.0        # logit_diff > 0  ↔  p_real > 0.5
BBOX_EXPAND         = 1.5        # expand face crop before inference (matches demo)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _expand_bbox(bbox, frame_shape, factor=BBOX_EXPAND):
    """Expand (x1,y1,x2,y2) around its centre by `factor`, clamped to frame."""
    x1, y1, x2, y2 = bbox
    h_frame, w_frame = frame_shape[:2]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw = (x2 - x1) * factor / 2
    hh = (y2 - y1) * factor / 2
    return (
        int(max(0, cx - hw)),
        int(max(0, cy - hh)),
        int(min(w_frame, cx + hw)),
        int(min(h_frame, cy + hh)),
    )


def _preprocess_rgb(face_rgb: np.ndarray) -> np.ndarray:
    """RGB crop → float32 NCHW blob.  /255 normalisation, no mean/std."""
    img = cv2.resize(face_rgb, INPUT_SIZE)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))   # HWC → CHW
    return np.expand_dims(img, 0)        # → NCHW


# ── Main class ─────────────────────────────────────────────────────────────────

class AntiSpoofChecker:
    """
    Liveness checker that matches the GitHub demo inference pipeline exactly.

    Parameters
    ----------
    model_path  : path to best_model_quantized.onnx or best_model.onnx
    threshold   : logit-difference threshold.  Default 0.0 = p_real > 50 %.
                  Raise (e.g. 0.5) to be stricter; lower to be more permissive.
    expand      : bbox expansion factor before cropping (default 1.5, same as demo)
    """

    def __init__(
        self,
        model_path: str = "best_model_quantized.onnx",
        threshold: float = DEFAULT_THRESHOLD,
        expand: float = BBOX_EXPAND,
    ):
        print(f"[AntiSpoof] Loading: {model_path}")
        self.session   = ort.InferenceSession(model_path)
        self.threshold = threshold
        self.expand    = expand
        print(f"[AntiSpoof] Ready  | input=128×128 RGB | logit threshold={threshold:.2f}")

    # ── Core inference ─────────────────────────────────────────────────────────

    def check(self, face_bgr: np.ndarray) -> tuple:
        """
        Run liveness on a BGR face crop (as returned by OpenCV).

        Returns
        -------
        (logit_diff: float, is_real: bool)
            logit_diff = real_logit - spoof_logit.
            Positive → real/live.  Negative → spoof.
        """
        if face_bgr is None or face_bgr.size == 0:
            return -999.0, False

        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        blob     = _preprocess_rgb(face_rgb)
        logits   = self.session.run([OUTPUT_NAME], {INPUT_NAME: blob})[0][0]
        diff     = float(logits[REAL_IDX] - logits[SPOOF_IDX])
        return diff, diff > self.threshold

    def check_from_frame(
        self,
        frame_bgr: np.ndarray,
        bbox: tuple,
    ) -> tuple:
        """
        Crop from a full frame using an expanded bbox, then run liveness.

        Parameters
        ----------
        frame_bgr : full BGR video frame
        bbox      : (x1, y1, x2, y2) face bounding box

        Returns
        -------
        (logit_diff: float, is_real: bool)
        """
        ex1, ey1, ex2, ey2 = _expand_bbox(bbox, frame_bgr.shape, self.expand)
        face_crop = frame_bgr[ey1:ey2, ex1:ex2]
        return self.check(face_crop)

    # ── Drawing helper ─────────────────────────────────────────────────────────

    def annotate_frame(
        self,
        frame: np.ndarray,
        bbox: tuple,
    ) -> tuple:
        """
        Crop (with expansion), run liveness, draw bbox + label on frame in-place.

        Parameters
        ----------
        frame : BGR video frame (modified in-place)
        bbox  : (x1, y1, x2, y2) face bbox from InsightFace

        Returns
        -------
        (is_real: bool, logit_diff: float)
        """
        x1, y1, x2, y2 = bbox
        diff, is_real   = self.check_from_frame(frame, bbox)

        color  = (0, 200, 0) if is_real else (0, 0, 255)
        status = "LIVE" if is_real else "SPOOF"
        label  = f"{status}: {diff:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame, label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65, color, 2,
        )
        return is_real, diff


# ── Standalone smoke-test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    model = sys.argv[2] if len(sys.argv) > 2 else "best_model_quantized.onnx"
    checker = AntiSpoofChecker(model)

    if len(sys.argv) < 2:
        print("No image supplied — random-tensor sanity check.")
        dummy = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
        diff, is_real = checker.check(dummy)
        print(f"logit_diff={diff:.4f}  → {'LIVE' if is_real else 'SPOOF'}")
        sys.exit(0)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"[Error] Cannot read: {sys.argv[1]}")
        sys.exit(1)

    diff, is_real = checker.check(img)
    print(f"logit_diff={diff:.4f}  → {'LIVE' if is_real else 'SPOOF'}")
