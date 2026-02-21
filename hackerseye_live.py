"""
hackerseye_live.py — Real-time face recognition + anti-spoofing

Unified pipeline:
  1. InsightFace detects faces and extracts recognition embeddings.
  2. (Optional) A trained anti-spoof CNN classifies each face crop
     as REAL or SPOOF.
  3. Only REAL faces proceed to recognition (compare against faces.db).
  4. SPOOF faces are drawn in red with a "SPOOF" label — identity hidden.

If no anti-spoof checkpoint is provided (--no-antispoof or missing file),
the system degrades gracefully to recognition-only mode (original behaviour).

Optimised for speed:
  - Smaller detection resolution (320×320)
  - Skip-frame detection: run heavy face model every N frames
  - Pre-normalise DB embeddings once at load time
  - Anti-spoof model runs only when a face is detected

Controls:
    Q     — quit
    R     — reload database
    S     — toggle stats overlay

Usage:
    # Recognition only (no anti-spoof — original behaviour)
    python hackerseye_live.py --no-antispoof

    # With anti-spoof model (Phase 3 baseline)
    python hackerseye_live.py --checkpoint checkpoints/best_antispoof.pt

    # With temporal anti-spoof model (Phase 4)
    python hackerseye_live.py --checkpoint checkpoints/best_temporal.pt --temporal
"""

import _nvidia_dll_fix  # noqa: F401 — register cuDNN/cuBLAS DLLs before ONNX

import argparse
import os
import sys
import cv2
import numpy as np
import time
import logging

from insightface.app import FaceAnalysis
from database import get_all_users, init_db

# ═══════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# CLI arguments
# ═══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="HackersEye — Live face recognition with anti-spoofing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── Anti-spoof model ─────────────────────────────────────
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to trained anti-spoof checkpoint (.pt). "
                        "If omitted, runs in recognition-only mode.")
    p.add_argument("--no-antispoof", action="store_true",
                   help="Disable anti-spoofing even if checkpoint exists.")
    p.add_argument("--temporal", action="store_true",
                   help="Use Phase 4 temporal anti-spoof model.")
    p.add_argument("--backbone", choices=["resnet18", "mobilenet"], default="resnet18")
    p.add_argument("--aggregator", choices=["avg", "max", "lstm"], default="lstm")
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--spoof-threshold", type=float, default=0.5,
                   help="Spoof probability threshold ( >= means spoof).")
    p.add_argument("--ema-alpha", type=float, default=0.3,
                   help="EMA smoothing factor for spoof predictions.")

    # ── Recognition ──────────────────────────────────────────
    p.add_argument("--threshold", type=float, default=0.45,
                   help="Cosine similarity threshold for face recognition.")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--det-size", type=int, default=320,
                   help="InsightFace detection resolution (square).")
    p.add_argument("--detect-every", type=int, default=3,
                   help="Run face detection every N frames.")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════
# Anti-spoof model loader
# ═══════════════════════════════════════════════════════════════

def load_antispoof_model(args, device):
    """
    Load the trained anti-spoof model from checkpoint.

    Returns (model, eval_transform, frame_buffer, smoother) or
    (None, None, None, None) if anti-spoof is disabled.
    """
    if args.no_antispoof or args.checkpoint is None:
        return None, None, None, None

    if not os.path.isfile(args.checkpoint):
        logger.warning(
            "Anti-spoof checkpoint not found: %s — running without anti-spoof.",
            args.checkpoint,
        )
        return None, None, None, None

    # Lazy imports — avoid loading PyTorch overhead in recognition-only mode
    from hackerseye_antispoof import (
        load_model, EVAL_TRANSFORM, FrameBuffer, PredictionSmoother,
    )

    # Build a lightweight args object for load_model compatibility
    class ModelArgs:
        temporal = args.temporal
        backbone = args.backbone
        aggregator = args.aggregator
        seq_len = args.seq_len
        checkpoint = args.checkpoint

    model = load_model(ModelArgs(), device)
    frame_buf = FrameBuffer(seq_len=args.seq_len)
    smoother = PredictionSmoother(alpha=args.ema_alpha, window=10)

    logger.info("Anti-spoof model loaded: %s", args.checkpoint)
    return model, EVAL_TRANSFORM, frame_buf, smoother


# ═══════════════════════════════════════════════════════════════
# Face recognition helpers
# ═══════════════════════════════════════════════════════════════

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalised vectors."""
    return float(np.dot(a, b))


def normalise(v: np.ndarray) -> np.ndarray:
    """L2-normalise a vector."""
    norm = np.linalg.norm(v)
    return v / (norm + 1e-8)


def load_users():
    """Load DB users and pre-normalise their embeddings."""
    raw_users = get_all_users()
    users = [(name, normalise(emb)) for name, emb in raw_users]
    logger.info("Loaded %d user(s): %s", len(users), [u[0] for u in users])
    return users


# ═══════════════════════════════════════════════════════════════
# Anti-spoof inference on a face crop
# ═══════════════════════════════════════════════════════════════

def run_antispoof_check(
    frame_bgr, bbox,
    model, transform, frame_buf, smoother,
    args, device,
):
    """
    Run anti-spoof inference on a detected face region.

    Returns (is_spoof: bool, spoof_confidence: float).
    Returns (False, 0.0) if anti-spoof is disabled (model is None).
    """
    import torch

    if model is None:
        return False, 0.0

    x1, y1, x2, y2 = bbox
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if (x2 - x1) < 20 or (y2 - y1) < 20:
        return False, 0.0

    # Crop face BGR → RGB, then apply eval transform → (3, 224, 224)
    face_rgb = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    face_tensor = transform(face_rgb)

    # Push into frame buffer (temporal model needs seq_len frames)
    frame_buf.push(face_tensor)

    raw_prob = None
    with torch.no_grad():
        if args.temporal and frame_buf.is_full:
            seq = frame_buf.get_sequence(device)
            logit = model(seq).squeeze()
            raw_prob = torch.sigmoid(logit).item()
        elif not args.temporal:
            inp = face_tensor.unsqueeze(0).to(device)
            logit = model(inp).squeeze()
            raw_prob = torch.sigmoid(logit).item()

    if raw_prob is not None:
        smoothed = smoother.update(raw_prob)
    else:
        smoothed = smoother.smoothed

    is_spoof = smoothed >= args.spoof_threshold
    confidence = smoothed if is_spoof else (1.0 - smoothed)
    return is_spoof, confidence


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── Anti-spoof model (optional) ──────────────────────────
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    as_model, as_transform, as_buf, as_smoother = load_antispoof_model(args, device)
    antispoof_on = as_model is not None

    # ── InsightFace recognition model ────────────────────────
    logger.info("Loading InsightFace model...")
    det_size = (args.det_size, args.det_size)
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=det_size)
    logger.info("InsightFace ready.  det_size=%s", det_size)

    # ── Database ─────────────────────────────────────────────
    init_db()
    users = load_users()
    if not users:
        logger.error("No users in database. Run: python register.py --name \"Name\"")
        sys.exit(1)

    # ── Webcam ───────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        logger.error("Cannot open camera %d.", args.camera)
        sys.exit(1)

    # ── State ────────────────────────────────────────────────
    fps_counter = 0
    fps_display = 0.0
    fps_timer = time.time()
    show_stats = True
    frame_num = 0
    cached_results = []   # list of (x1,y1,x2,y2, label, score, color, is_spoof)

    mode_str = "Recognition + Anti-Spoof" if antispoof_on else "Recognition only"
    logger.info("=" * 55)
    logger.info("HackersEye Live — %s", mode_str)
    logger.info("  Q=quit  R=reload DB  S=stats")
    if not antispoof_on:
        logger.warning("Anti-spoof DISABLED. Pass --checkpoint to enable.")
    logger.info("=" * 55)

    # ══════════════════════════════════════════════════════════
    # Main loop
    # ══════════════════════════════════════════════════════════
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # ── Run detection every N-th frame ───────────────────
        if frame_num % args.detect_every == 1 or args.detect_every == 1:
            faces = app.get(frame)
            cached_results = []

            for face in faces:
                x1, y1, x2, y2 = map(int, face.bbox)
                embedding = normalise(face.embedding.astype(np.float32))

                # ── Anti-spoof gate ──────────────────────────
                is_spoof = False
                spoof_conf = 0.0
                if antispoof_on:
                    is_spoof, spoof_conf = run_antispoof_check(
                        frame, (x1, y1, x2, y2),
                        as_model, as_transform, as_buf, as_smoother,
                        args, device,
                    )

                if is_spoof:
                    # SPOOF — do NOT reveal identity
                    cached_results.append(
                        (x1, y1, x2, y2, "SPOOF DETECTED", spoof_conf,
                         (0, 0, 255), True)
                    )
                else:
                    # REAL — proceed with recognition
                    best_name, best_score = "Unknown", 0.0
                    for name, db_emb in users:
                        score = cosine_similarity(embedding, db_emb)
                        if score > best_score:
                            best_score = score
                            best_name = name

                    if best_score < args.threshold:
                        best_name = "Unknown"
                        color = (0, 165, 255)   # Orange for unknown
                    else:
                        color = (0, 255, 0)     # Green for recognised

                    cached_results.append(
                        (x1, y1, x2, y2, best_name, best_score,
                         color, False)
                    )

        # ── Draw cached results on every frame ───────────────
        for (x1, y1, x2, y2, label, score, color, spoofed) in cached_results:
            thickness = 3 if spoofed else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            if spoofed:
                text = f"SPOOF  {score:.0%}"
            else:
                text = f"{label}  {score:.2f}"

            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            label_y = max(y1 - 10, th + 10)
            cv2.rectangle(frame,
                          (x1, label_y - th - 6),
                          (x1 + tw + 6, label_y + 2),
                          color, -1)
            cv2.putText(frame, text,
                        (x1 + 3, label_y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 0, 0), 2)

            if spoofed:
                cv2.putText(frame, "!! SPOOF ATTEMPT !!",
                            (x1, y2 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)

        # ── Stats overlay ────────────────────────────────────
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            fps_display = fps_counter / (time.time() - fps_timer)
            fps_counter = 0
            fps_timer = time.time()

        if show_stats:
            lines = [
                f"FPS:        {fps_display:.1f}",
                f"Faces:      {len(cached_results)}",
                f"Users:      {len(users)}",
                f"Rec thr:    {args.threshold}",
                f"Anti-spoof: {'ON' if antispoof_on else 'OFF'}",
            ]
            if antispoof_on:
                lines.append(f"Spoof thr:  {args.spoof_threshold}")
            for i, line in enumerate(lines):
                cv2.putText(frame, line,
                            (10, 22 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(frame, "Q=Quit  R=Reload  S=Stats",
                    (10, args.height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (180, 180, 180), 1, cv2.LINE_AA)

        cv2.imshow("HackersEye Live", frame)

        # ── Key handling ─────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            logger.info("Quit by user.")
            break
        elif key == ord('r'):
            logger.info("Reloading user database...")
            users = load_users()
        elif key == ord('s'):
            show_stats = not show_stats

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
