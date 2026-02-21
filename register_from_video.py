"""
register_from_video.py — Auto-register a face from a video file or webcam clip.

No spacebar needed. Frames are sampled automatically, faces detected,
embeddings collected and averaged, then saved to the database.

Usage:
    # From a video file:
    python register_from_video.py --name "Alice" --video alice.mp4

    # From webcam (records N seconds automatically):
    python register_from_video.py --name "Alice" --webcam --duration 5

    # Control how many samples to collect:
    python register_from_video.py --name "Alice" --video alice.mp4 --samples 20
"""

import cv2
import numpy as np
import argparse
import time
from insightface.app import FaceAnalysis
from database import init_db, add_user

# ──────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Auto-register a face from video.")
parser.add_argument("--name",     type=str,  required=True, help="Person's name")
parser.add_argument("--video",    type=str,  default=None,  help="Path to video file")
parser.add_argument("--webcam",   action="store_true",      help="Use webcam as source")
parser.add_argument("--duration", type=int,  default=5,     help="Webcam capture duration in seconds (default: 5)")
parser.add_argument("--samples",  type=int,  default=15,    help="Target number of clean embeddings to collect (default: 15)")
parser.add_argument("--skip",     type=int,  default=5,     help="Process every Nth frame (default: 5, lower = more frames checked)")
parser.add_argument("--ctx",      type=int,  default=0,     help="InsightFace ctx_id: 0=GPU, -1=CPU")
args = parser.parse_args()

if not args.video and not args.webcam:
    parser.error("Provide either --video <file> or --webcam")

# ──────────────────────────────────────────────
# Init
# ──────────────────────────────────────────────
init_db()
print(f"\n[Register] Name    : {args.name}")
print(f"[Register] Source  : {'Webcam (' + str(args.duration) + 's)' if args.webcam else args.video}")
print(f"[Register] Samples : {args.samples} embeddings to collect\n")

print("[Init] Loading InsightFace model...")
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=args.ctx)
print("[Init] Model ready.\n")

# ──────────────────────────────────────────────
# Open video source
# ──────────────────────────────────────────────
if args.webcam:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print(f"[Webcam] Recording for {args.duration} seconds... look at the camera!")
    time.sleep(1)  # short pause so user can get ready
    deadline = time.time() + args.duration
else:
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[Error] Cannot open video: {args.video}")
        exit(1)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video    = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"[Video] {total_frames} frames @ {fps_video:.1f} fps")
    deadline = None

# ──────────────────────────────────────────────
# Frame extraction loop
# ──────────────────────────────────────────────
embeddings_collected = []
frame_idx = 0
checked   = 0

while True:
    # Stop webcam after duration
    if args.webcam and time.time() > deadline:
        print("\n[Webcam] Time's up.")
        break

    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    # Skip frames for speed
    if frame_idx % args.skip != 0:
        continue

    checked += 1
    faces = face_app.get(frame)

    # Only use frames with exactly one face (avoids confusion)
    if len(faces) != 1:
        status = "no face" if len(faces) == 0 else "multiple faces"
        print(f"  Frame {frame_idx:>5} — skipped ({status})")
        continue

    face  = faces[0]
    det_score = float(face.det_score)

    # Reject low-confidence detections (blur, angle, partial face)
    if det_score < 0.75:
        print(f"  Frame {frame_idx:>5} — skipped (low confidence: {det_score:.2f})")
        continue

    emb = face.embedding.astype(np.float32)
    emb /= np.linalg.norm(emb)   # L2-normalise
    embeddings_collected.append(emb)
    print(f"  Frame {frame_idx:>5} — ✔ captured  (conf={det_score:.2f}, total={len(embeddings_collected)}/{args.samples})")

    if len(embeddings_collected) >= args.samples:
        print(f"\n[Done] Reached {args.samples} samples.")
        break

cap.release()

# ──────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────
print(f"\n[Result] Collected {len(embeddings_collected)} clean embeddings from {checked} checked frames.")

if len(embeddings_collected) == 0:
    print("❌ No usable faces found. Try a clearer video with good lighting and a frontal view.")
    exit(1)

if len(embeddings_collected) < 3:
    print(f"⚠️  Only {len(embeddings_collected)} sample(s) — result may be less accurate. Consider a longer/clearer video.")

avg_embedding = np.mean(embeddings_collected, axis=0)
avg_embedding /= np.linalg.norm(avg_embedding)   # re-normalise after averaging

add_user(args.name, avg_embedding)
print(f"\n✅ '{args.name}' registered successfully using {len(embeddings_collected)} averaged samples!")
print("   Run live_recognition.py to start recognising.\n")
