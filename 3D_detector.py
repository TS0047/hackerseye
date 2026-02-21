import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.signal import butter, filtfilt
from collections import deque

# ---------------- CONFIG ----------------
FPS = 30
WINDOW_SIZE = 180          
PULSE_THRESHOLD = 0.015
RIGIDITY_THRESHOLD = 0.6
MODEL_PATH = 'face_landmarker.task' # Ensure this file is in your directory
# ----------------------------------------

# Initialize MediaPipe Tasks Face Landmarker
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO, # Optimized for camera/video streams
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.FaceLandmarker.create_from_options(options)

green_signal = deque(maxlen=WINDOW_SIZE)
prev_landmarks = None

def bandpass_filter(signal, fs=30):
    nyq = 0.5 * fs
    low = 0.7 / nyq
    high = 4.0 / nyq
    b, a = butter(3, [low, high], btype='band')
    return filtfilt(b, a, signal)

def extract_cheek_roi(frame, landmarks):
    h, w, _ = frame.shape
    # Approximate cheek indices in new Face Mesh (234, 454 are boundary points)
    cheek_ids = [234, 454, 93, 323]
    
    xs = [int(lm.x * w) for lm in landmarks]
    ys = [int(lm.y * h) for lm in landmarks]
    
    # Simple bounding box from cheek-related indices
    try:
        x_min, x_max = min([xs[i] for i in cheek_ids]), max([xs[i] for i in cheek_ids])
        y_min, y_max = min([ys[i] for i in cheek_ids]), max([ys[i] for i in cheek_ids])
        return frame[y_min:y_max, x_min:x_max]
    except:
        return None

def compute_rigidity(prev_pts, curr_pts):
    diffs = np.linalg.norm(curr_pts - prev_pts, axis=1)
    return np.std(diffs)

cap = cv2.VideoCapture(0)
print("Press ESC to exit")

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to MediaPipe Image object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Calculate timestamp in milliseconds
    timestamp_ms = int((frame_count / FPS) * 1000)
    detection_result = detector.detect_for_video(mp_image, timestamp_ms)
    frame_count += 1

    verdict = "NO FACE"
    color = (255, 255, 255)

    if detection_result.face_landmarks:
        landmarks = detection_result.face_landmarks[0]

        # ----- rPPG (Pulse) -----
        roi = extract_cheek_roi(frame, landmarks)
        if roi is not None and roi.size > 0:
            green_mean = np.mean(roi[:, :, 1])
            green_signal.append(green_mean)

        pulse_energy = 0.0
        pulse_ok = False
        if len(green_signal) == WINDOW_SIZE:
            filtered = bandpass_filter(np.array(green_signal), FPS)
            pulse_energy = np.std(filtered)
            pulse_ok = pulse_energy > PULSE_THRESHOLD

        # ----- Motion Rigidity -----
        curr_pts = np.array([(lm.x, lm.y, lm.z) for lm in landmarks], dtype=np.float32)
        rigidity = 0.0
        motion_ok = True

        if prev_landmarks is not None:
            rigidity = compute_rigidity(prev_landmarks, curr_pts)
            motion_ok = rigidity > RIGIDITY_THRESHOLD
        prev_landmarks = curr_pts

        # ----- Decision -----
        if pulse_ok and motion_ok:
            verdict, color = "REAL FACE", (0, 255, 0)
        else:
            verdict, color = "SPOOF DETECTED", (0, 0, 255)

        # Display Data
        cv2.putText(frame, verdict, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Pulse: {pulse_energy:.4f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.putText(frame, f"Rigidity: {rigidity:.4f}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    cv2.imshow("3D Mask Spoof Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
