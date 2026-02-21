"""
live_recognition.py — Real-time face recognition using InsightFace + SQLite
                       with liveness / anti-spoof detection (MiniFASNet ONNX)

Controls:
    Q     — quit
    R     — reload database (pick up newly registered users without restarting)
    S     — toggle stats overlay
    L     — toggle liveness check on/off
"""

import cv2
import numpy as np
import time
from insightface.app import FaceAnalysis
from database import get_all_users, init_db
from anti_spoof_utils import AntiSpoofChecker

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
THRESHOLD    = 0.45   # Cosine similarity threshold (lower = stricter)
FRAME_W      = 640
FRAME_H      = 480
MODEL_NAME   = "buffalo_l"
CTX_ID       = 0      # 0 = GPU, -1 = CPU

# Anti-spoof
SPOOF_MODEL   = "best_model_quantized.onnx"   # swap for best_model.onnx if needed
SPOOF_THRESH  = 0.50        # logit-diff cutoff: 0.0 = p_real>50%, raise to be stricter

# ──────────────────────────────────────────────
# Load model
# ──────────────────────────────────────────────
print("[Init] Loading InsightFace model...")
app = FaceAnalysis(name=MODEL_NAME)
app.prepare(ctx_id=CTX_ID)
print("[Init] Model ready.")

print("[Init] Loading anti-spoof model...")
spoof_checker  = AntiSpoofChecker(SPOOF_MODEL, threshold=SPOOF_THRESH)
liveness_on    = True   # toggled with L key

# ──────────────────────────────────────────────
# Similarity
# ──────────────────────────────────────────────
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))

# ──────────────────────────────────────────────
# Load users
# ──────────────────────────────────────────────
def load_users():
    users = get_all_users()
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

print("\n✅ Live recognition started.")
print("   Q = quit | R = reload DB | S = toggle stats | L = toggle liveness\n")

# ──────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        print("[Error] Frame capture failed.")
        break

    # ── Detect & recognise ──────────────────────
    faces = app.get(frame)

    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)

        # ── Liveness check ──────────────────────────
        if liveness_on:
            face_crop = frame[y1:y2, x1:x2]
            is_live, spoof_score = spoof_checker.annotate_frame(
                frame, (x1, y1, x2, y2)
            )
            if not is_live:
                continue   # spoof detected — skip recognition entirely

        embedding = face.embedding.astype(np.float32)

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
            f"FPS:     {fps_display:.1f}",
            f"Faces:   {len(faces)}",
            f"Users:   {len(users)}",
            f"Thr:     {THRESHOLD}",
            f"Spoof:   {'ON' if liveness_on else 'OFF'}",
        ]
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line,
                        (10, 22 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(frame, "Q=Quit  R=Reload  S=Stats  L=Liveness",
                (10, FRAME_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (180, 180, 180), 1, cv2.LINE_AA)

    cv2.imshow("Live Face Recognition", frame)

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

    elif key == ord('l'):
        liveness_on = not liveness_on
        state = "ON" if liveness_on else "OFF"
        print(f"[Liveness] Anti-spoof toggled {state}")

cap.release()
cv2.destroyAllWindows()
