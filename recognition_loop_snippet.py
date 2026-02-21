"""
recognition_loop_snippet.py
───────────────────────────
Drop-in additions for your main face-recognition script.
Search for each "STEP" comment and add the code shown.
"""

# ════════════════════════════════════════════════════════════════════
# STEP 1 — Import (top of your script, alongside other imports)
# ════════════════════════════════════════════════════════════════════
from anti_spoof_utils import AntiSpoofChecker

# ════════════════════════════════════════════════════════════════════
# STEP 2 — Initialise once, after InsightFace loads
# ════════════════════════════════════════════════════════════════════
spoof_checker = AntiSpoofChecker(
    model_path="anti_spoof.onnx",
    threshold=0.7,          # raise to 0.8 in well-lit environments
)

# ════════════════════════════════════════════════════════════════════
# STEP 3 — Inside your `for face in faces:` loop
#           Replace your existing bbox / recognition block with this
# ════════════════════════════════════════════════════════════════════

# --- Example loop structure (adapt variable names to match yours) ---
# for face in faces:
#
#     bbox = face.bbox.astype(int)          # [x1, y1, x2, y2]
#     x1, y1, x2, y2 = bbox
#
#     # ── Liveness check ────────────────────────────────────────────
#     face_crop = frame[y1:y2, x1:x2]
#
#     is_live, spoof_score = spoof_checker.annotate_frame(
#         frame, face_crop, (x1, y1, x2, y2)
#     )
#
#     if not is_live:
#         continue                          # skip recognition for spoofs
#
#     # ── Recognition (your existing code below this line) ──────────
#     embedding = face.normed_embedding
#     name, confidence = match_face(embedding)   # your matching function
#     ...

# ════════════════════════════════════════════════════════════════════
# MINIMAL inline version (if you prefer not to use annotate_frame)
# ════════════════════════════════════════════════════════════════════

# face_crop = frame[y1:y2, x1:x2]
# if face_crop.size == 0:
#     continue
#
# real_score = spoof_checker.check(face_crop)
#
# if real_score < spoof_checker.threshold:
#     color = (0, 0, 255)
#     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#     cv2.putText(frame, f"SPOOF {real_score:.2f}",
#                 (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
#     continue
#
# # → live face, proceed with recognition
