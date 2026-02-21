import cv2
import numpy as np
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.signal import butter, filtfilt
from collections import deque

FPS = 30
WINDOW_SIZE = 180
PULSE_THRESHOLD = 0.015
RIGIDITY_LOWER = 0.0001
RIGIDITY_UPPER = 0.05
RIGIDITY_HISTORY = 30
MODEL_PATH = 'face_landmarker.task'

CHEEK_IDS = [234, 454, 93, 323]

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.FaceLandmarker.create_from_options(options)

green_signal = deque(maxlen=WINDOW_SIZE)
rigidity_history = deque(maxlen=RIGIDITY_HISTORY)
prev_landmarks = None
start_time = time.time()


def bandpass_filter(signal, fs=30):
    nyq = 0.5 * fs
    low = 0.7 / nyq
    high = 4.0 / nyq
    b, a = butter(3, [low, high], btype='band')
    return filtfilt(b, a, signal)


def extract_cheek_roi(frame, landmarks):
    h, w, _ = frame.shape
    xs = [int(landmarks[i].x * w) for i in CHEEK_IDS]
    ys = [int(landmarks[i].y * h) for i in CHEEK_IDS]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    roi = frame[y_min:y_max, x_min:x_max]
    return roi if roi.size > 0 else None


def compute_rigidity(prev_pts, curr_pts):
    diffs = np.linalg.norm(curr_pts - prev_pts, axis=1)
    return np.std(diffs)


def draw_overlay(frame, verdict, color, pulse_energy, avg_rigidity, calibrating):
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, 20), (400, 155), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    if calibrating:
        elapsed = time.time() - start_time
        progress = min(elapsed / (WINDOW_SIZE / FPS), 1.0)
        bar_width = int(360 * progress)
        cv2.rectangle(frame, (20, 20), (380, 58), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, 20), (20 + bar_width, 58), (200, 180, 0), -1)
        cv2.putText(frame, f"Calibrating... {int(progress * 100)}%", (30, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    else:
        cv2.putText(frame, verdict, (30, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)

    cv2.putText(frame, f"Pulse Energy : {pulse_energy:.5f}  (need > {PULSE_THRESHOLD})", (30, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
    cv2.putText(frame, f"Avg Rigidity : {avg_rigidity:.5f}  (need {RIGIDITY_LOWER} - {RIGIDITY_UPPER})", (30, 118),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)


cap = cv2.VideoCapture(0)
print("Press ESC to exit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    timestamp_ms = int((time.time() - start_time) * 1000)
    detection_result = detector.detect_for_video(mp_image, timestamp_ms)

    if not detection_result.face_landmarks:
        cv2.putText(frame, "NO FACE DETECTED", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Face Spoof Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    landmarks = detection_result.face_landmarks[0]
    curr_pts = np.array([(lm.x, lm.y, lm.z) for lm in landmarks], dtype=np.float32)

    roi = extract_cheek_roi(frame, landmarks)
    if roi is not None:
        green_signal.append(np.mean(roi[:, :, 1]))

    if prev_landmarks is not None:
        r = compute_rigidity(prev_landmarks, curr_pts)
        rigidity_history.append(r)

    prev_landmarks = curr_pts

    avg_rigidity = float(np.mean(rigidity_history)) if rigidity_history else 0.0
    motion_ok = RIGIDITY_LOWER < avg_rigidity < RIGIDITY_UPPER

    pulse_energy = 0.0
    pulse_ok = False
    calibrating = len(green_signal) < WINDOW_SIZE

    if not calibrating:
        filtered = bandpass_filter(np.array(green_signal), FPS)
        pulse_energy = np.std(filtered)
        pulse_ok = pulse_energy > PULSE_THRESHOLD

    if calibrating:
        verdict, color = "CALIBRATING", (200, 180, 0)
    elif pulse_ok and motion_ok:
        verdict, color = "REAL FACE", (0, 220, 0)
    elif avg_rigidity <= RIGIDITY_LOWER:
        verdict, color = "SPOOF: STATIC IMAGE", (0, 0, 255)
    elif not pulse_ok:
        verdict, color = "SPOOF: NO PULSE", (0, 80, 255)
    else:
        verdict, color = "SPOOF DETECTED", (0, 0, 255)

    draw_overlay(frame, verdict, color, pulse_energy, avg_rigidity, calibrating)
    cv2.imshow("Face Spoof Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()