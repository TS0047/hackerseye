"""
hackerseye_live.py — Real-time face recognition using InsightFace + SQLite

Optimised for speed:
  - Smaller detection resolution (320×320) for faster face detection
  - Skip-frame detection: run heavy face model every N frames,
    reuse cached bounding boxes + embeddings in between
  - Pre-normalise DB embeddings once at load time

Controls:
    Q     — quit
    R     — reload database (pick up newly registered users without restarting)
    S     — toggle stats overlay
"""

import _nvidia_dll_fix  # noqa: F401 — register cuDNN/cuBLAS DLLs before ONNX

import cv2
import numpy as np
import time
from insightface.app import FaceAnalysis
from database import get_all_users, init_db

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
THRESHOLD    = 0.45    # Cosine similarity threshold (lower = stricter)
FRAME_W      = 640
FRAME_H      = 480
MODEL_NAME   = "buffalo_l"
CTX_ID       = 0       # 0 = GPU (ONNX Runtime CUDA), -1 = CPU
DET_SIZE     = (320, 320)  # Detection resolution (smaller = faster; default was 640×640)
DETECT_EVERY = 3       # Run face detection every N frames (reuse boxes in between)

# ──────────────────────────────────────────────
# Load model — use smaller det_size for speed
# ──────────────────────────────────────────────
print("[Init] Loading InsightFace model...")
app = FaceAnalysis(name=MODEL_NAME)
app.prepare(ctx_id=CTX_ID, det_size=DET_SIZE)
print(f"[Init] Model ready.  det_size={DET_SIZE}  ctx={'GPU' if CTX_ID >= 0 else 'CPU'}")

# ──────────────────────────────────────────────
# Similarity — vectorised for multiple DB users
# ──────────────────────────────────────────────
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalised vectors."""
    return float(np.dot(a, b))


def normalise(v: np.ndarray) -> np.ndarray:
    """L2-normalise a vector (or return zero-vector if norm ≈ 0)."""
    norm = np.linalg.norm(v)
    return v / (norm + 1e-8)

# ──────────────────────────────────────────────
# Load users (pre-normalise embeddings once)
# ──────────────────────────────────────────────
def load_users():
    """Load DB users and pre-normalise their embeddings for fast cosine sim."""
    raw_users = get_all_users()
    users = [(name, normalise(emb)) for name, emb in raw_users]
    print(f"[DB] Loaded {len(users)} user(s): {[u[0] for u in users]}")
    return users

init_db()
users = load_users()

if not users:
    print("\n⚠️  No users in database. Run register.py first:\n")
    print("    python register.py --name \"Your Name\"\n")
    exit(1)

# ──────────────────────────────────────────────
# Open webcam
# ──────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("[Error] Cannot open webcam.")
    exit(1)

# ──────────────────────────────────────────────
# FPS tracking
# ──────────────────────────────────────────────
fps_counter  = 0
fps_display  = 0.0
fps_timer    = time.time()
show_stats   = True
frame_num    = 0

# Cache for skip-frame reuse
cached_results = []   # list of (x1,y1,x2,y2, best_name, best_score, color)

print("\n✅ HackersEye Live started.")
print(f"   Detection every {DETECT_EVERY} frames | det_size={DET_SIZE}")
print("   Q = quit | R = reload DB | S = toggle stats\n")

# ──────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        print("[Error] Frame capture failed.")
        break

    frame_num += 1

    # ── Run detection only every N-th frame ─────
    if frame_num % DETECT_EVERY == 1 or DETECT_EVERY == 1:
        faces = app.get(frame)

        cached_results = []
        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            embedding = normalise(face.embedding.astype(np.float32))

            best_name  = "Unknown"
            best_score = 0.0

            for name, db_emb in users:
                score = cosine_similarity(embedding, db_emb)
                if score > best_score:
                    best_score = score
                    best_name  = name

            if best_score < THRESHOLD:
                best_name = "Unknown"
                color     = (0, 0, 255)   # Red
            else:
                color     = (0, 255, 0)   # Green

            cached_results.append((x1, y1, x2, y2, best_name, best_score, color))

    # ── Draw cached results on every frame ──────
    for (x1, y1, x2, y2, best_name, best_score, color) in cached_results:
        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background
        label  = f"{best_name}  {best_score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        label_y = max(y1 - 10, th + 10)
        cv2.rectangle(frame,
                      (x1, label_y - th - 6),
                      (x1 + tw + 6, label_y + 2),
                      color, -1)
        cv2.putText(frame, label,
                    (x1 + 3, label_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 0),   # Black text on coloured bg
                    2)

    # ── Stats overlay ───────────────────────────
    fps_counter += 1
    if time.time() - fps_timer >= 1.0:
        fps_display  = fps_counter / (time.time() - fps_timer)
        fps_counter  = 0
        fps_timer    = time.time()

    if show_stats:
        info_lines = [
            f"FPS:   {fps_display:.1f}",
            f"Faces: {len(cached_results)}",
            f"Users: {len(users)}",
            f"Thr:   {THRESHOLD}",
        ]
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line,
                        (10, 22 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(frame, "Q=Quit  R=Reload  S=Stats",
                (10, FRAME_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (180, 180, 180), 1, cv2.LINE_AA)

    cv2.imshow("HackersEye Live Recognition", frame)

    # ── Key handling ────────────────────────────
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("[Exit] Quit by user.")
        break

    elif key == ord('r'):
        print("[Reload] Reloading user database...")
        users = load_users()

    elif key == ord('s'):
        show_stats = not show_stats

cap.release()
cv2.destroyAllWindows()
