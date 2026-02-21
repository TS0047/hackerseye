"""
register.py — Capture face from webcam and register into the database.

Usage:
    python register.py --name "Your Name" [--samples 5]

Controls (during capture):
    SPACE  — capture current frame
    Q      — quit without saving
"""

import _nvidia_dll_fix  # noqa: F401 — register cuDNN/cuBLAS DLLs before ONNX

import cv2
import numpy as np
import argparse
from insightface.app import FaceAnalysis
from database import init_db, add_user

# ──────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Register a face into the database.")
parser.add_argument("--name", type=str, required=True, help="Person's name")
parser.add_argument("--samples", type=int, default=5,
                    help="Number of face samples to capture and average (default: 5)")
args = parser.parse_args()

# ──────────────────────────────────────────────
# Init
# ──────────────────────────────────────────────
init_db()

print(f"\n[Register] Registering: {args.name}")
print(f"[Register] Will capture {args.samples} samples and average them for robustness.\n")

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)  # ctx_id=0 → GPU, -1 → CPU

# ──────────────────────────────────────────────
# Open webcam
# ──────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

embeddings_collected = []
print("Press SPACE to capture a sample. Press Q to quit.")

while len(embeddings_collected) < args.samples:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    faces = app.get(frame)

    if len(faces) == 0:
        cv2.putText(display, "No face detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    elif len(faces) > 1:
        cv2.putText(display, "Multiple faces! Show only one.", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 165, 255), 2)
    else:
        face = faces[0]
        x1, y1, x2, y2 = map(int, face.bbox)
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display, f"SPACE to capture ({len(embeddings_collected)}/{args.samples})",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(display, f"Registering: {args.name}", (20, display.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

    cv2.imshow("Register Face", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("\n[Register] Cancelled.")
        cap.release()
        cv2.destroyAllWindows()
        exit()

    if key == ord(' ') and len(faces) == 1:
        emb = faces[0].embedding.astype(np.float32)
        embeddings_collected.append(emb)
        print(f"  ✔ Sample {len(embeddings_collected)}/{args.samples} captured.")

cap.release()
cv2.destroyAllWindows()

# ──────────────────────────────────────────────
# Average embeddings & save
# ──────────────────────────────────────────────
if len(embeddings_collected) == args.samples:
    avg_embedding = np.mean(embeddings_collected, axis=0)
    # L2-normalize for cosine similarity
    avg_embedding /= np.linalg.norm(avg_embedding)
    add_user(args.name, avg_embedding)
    print(f"\n✅ '{args.name}' registered successfully with {args.samples} samples!")
else:
    print("\n[Register] Not enough samples. Registration aborted.")
