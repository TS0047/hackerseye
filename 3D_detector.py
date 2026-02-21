import cv2
import numpy as np
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.signal import butter, filtfilt, welch
from collections import deque

FPS              = 30
WINDOW_SIZE      = 180
RPPG_THRESHOLD   = 0.015
FREQ_SNR_MIN     = 2.5
RIGIDITY_LOWER   = 0.0001
RIGIDITY_UPPER   = 0.05
RIGIDITY_HISTORY = 30
DEPTH_VAR_MIN    = 0.002
BLINK_EAR_THRESH = 0.21
BLINK_CONSEC     = 2
LBP_REAL_MAX     = 0.55
MODEL_PATH       = 'face_landmarker.task'
CHEEK_IDS        = [234, 454, 93, 323]

LEFT_EYE_IDS  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDS = [33,  160, 158, 133, 153, 144]

WEIGHTS = {
    'pulse':    0.30,
    'motion':   0.20,
    'depth':    0.20,
    'blink':    0.15,
    'texture':  0.15,
}

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.FaceLandmarker.create_from_options(options)

green_signal     = deque(maxlen=WINDOW_SIZE)
rigidity_history = deque(maxlen=RIGIDITY_HISTORY)
blink_counter    = 0
blink_total      = 0
consec_below     = 0
prev_landmarks   = None
start_time       = time.time()


def bandpass_filter(signal, fs=30):
    nyq  = 0.5 * fs
    low  = 0.7 / nyq
    high = 4.0 / nyq
    b, a = butter(3, [low, high], btype='band')
    return filtfilt(b, a, signal)


def pulse_snr(signal, fs=30):
    freqs, psd = welch(signal, fs=fs, nperseg=min(len(signal), 256))
    mask       = (freqs >= 0.7) & (freqs <= 4.0)
    band_psd   = psd[mask]
    if band_psd.sum() == 0:
        return 0.0
    peak_power = band_psd.max()
    noise_power = (band_psd.sum() - peak_power) / (len(band_psd) - 1 + 1e-8)
    return float(peak_power / (noise_power + 1e-8))


def extract_cheek_roi(frame, landmarks):
    h, w, _ = frame.shape
    xs  = [int(landmarks[i].x * w) for i in CHEEK_IDS]
    ys  = [int(landmarks[i].y * h) for i in CHEEK_IDS]
    roi = frame[min(ys):max(ys), min(xs):max(xs)]
    return roi if roi.size > 0 else None


def compute_rigidity(prev_pts, curr_pts):
    diffs = np.linalg.norm(curr_pts - prev_pts, axis=1)
    return np.std(diffs)


def eye_aspect_ratio(landmarks, eye_ids, w, h):
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in eye_ids])
    A   = np.linalg.norm(pts[1] - pts[5])
    B   = np.linalg.norm(pts[2] - pts[4])
    C   = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C + 1e-8)


def depth_variance(landmarks):
    z_vals = np.array([lm.z for lm in landmarks])
    return float(np.var(z_vals))


def lbp_uniformity(roi):
    if roi is None or roi.size == 0:
        return 0.5
    gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    gray    = cv2.resize(gray, (64, 64))
    lbp     = np.zeros_like(gray, dtype=np.uint8)
    padded  = np.pad(gray, 1, mode='reflect').astype(np.int16)
    center  = padded[1:-1, 1:-1].astype(np.int16)
    neighbors = [
        padded[0:-2, 0:-2], padded[0:-2, 1:-1], padded[0:-2, 2:],
        padded[1:-1, 2:],   padded[2:,   2:],   padded[2:,   1:-1],
        padded[2:,   0:-2], padded[1:-1, 0:-2],
    ]
    for i, nb in enumerate(neighbors):
        lbp |= ((nb >= center).astype(np.uint8) << i)
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256), density=True)
    entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0] + 1e-8))
    return float(entropy / 8.0)


def compute_scores(pulse_energy, snr, avg_rigidity, depth_var,
                   blink_total, elapsed, lbp_score, calibrating):
    scores = {}

    if calibrating:
        return {k: 0.5 for k in WEIGHTS}

    pulse_ok   = pulse_energy > RPPG_THRESHOLD and snr > FREQ_SNR_MIN
    scores['pulse'] = 1.0 if pulse_ok else max(0.0, min(pulse_energy / RPPG_THRESHOLD, 1.0) * 0.5)

    motion_ok = RIGIDITY_LOWER < avg_rigidity < RIGIDITY_UPPER
    if motion_ok:
        scores['motion'] = 1.0
    elif avg_rigidity <= RIGIDITY_LOWER:
        scores['motion'] = 0.0
    else:
        scores['motion'] = 0.3

    depth_ok = depth_var > DEPTH_VAR_MIN
    scores['depth'] = min(depth_var / (DEPTH_VAR_MIN * 3), 1.0) if depth_ok else depth_var / DEPTH_VAR_MIN * 0.5

    expected_blinks = (elapsed / 60.0) * 15
    blink_ratio     = blink_total / (expected_blinks + 1e-8)
    scores['blink'] = min(blink_ratio, 1.0) if elapsed > 10 else 0.5

    scores['texture'] = 1.0 - lbp_score if lbp_score < LBP_REAL_MAX else 0.2

    return scores


def weighted_verdict(scores):
    total = sum(WEIGHTS[k] * scores[k] for k in WEIGHTS)
    return total


def get_face_crop(frame, landmarks):
    h, w, _ = frame.shape
    xs  = [int(lm.x * w) for lm in landmarks]
    ys  = [int(lm.y * h) for lm in landmarks]
    pad = 20
    x1  = max(min(xs) - pad, 0)
    x2  = min(max(xs) + pad, w)
    y1  = max(min(ys) - pad, 0)
    y2  = min(max(ys) + pad, h)
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def draw_overlay(frame, verdict, color, scores, metrics, calibrating, progress):
    h_f, w_f = frame.shape[:2]
    overlay  = frame.copy()
    cv2.rectangle(overlay, (15, 15), (430, 310), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    cv2.rectangle(frame, (15, 15), (430, 310), color, 2)

    if calibrating:
        bar_w = int(395 * progress)
        cv2.rectangle(frame, (15, 15), (410, 55), (40, 40, 40), -1)
        cv2.rectangle(frame, (15, 15), (15 + bar_w, 55), (180, 155, 0), -1)
        cv2.putText(frame, f"Calibrating...  {int(progress * 100)}%",
                    (25, 43), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1)
    else:
        cv2.putText(frame, verdict, (25, 55),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)

    score_rows = [
        ("Pulse / rPPG",  scores.get('pulse',   0.5)),
        ("Motion",        scores.get('motion',  0.5)),
        ("Depth (3D)",    scores.get('depth',   0.5)),
        ("Blink rate",    scores.get('blink',   0.5)),
        ("Skin texture",  scores.get('texture', 0.5)),
    ]
    for i, (label, val) in enumerate(score_rows):
        y         = 88 + i * 34
        bar_full  = 170
        bar_fill  = int(bar_full * val)
        bar_color = (0, int(210 * val), int(210 * (1 - val)))
        cv2.rectangle(frame, (210, y - 13), (210 + bar_full, y + 6), (50, 50, 50), -1)
        cv2.rectangle(frame, (210, y - 13), (210 + bar_fill,  y + 6), bar_color, -1)
        cv2.putText(frame, label, (25, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (190, 190, 190), 1)
        cv2.putText(frame, f"{val:.2f}", (388, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1)

    cv2.line(frame, (15, 265), (430, 265), (55, 55, 55), 1)
    meta = [
        f"Pulse energy: {metrics['pulse_e']:.4f}   SNR: {metrics['snr']:.1f}",
        f"Rigidity: {metrics['rigidity']:.5f}   Depth var: {metrics['depth_v']:.5f}   Blinks: {metrics['blinks']}",
    ]
    for i, line in enumerate(meta):
        cv2.putText(frame, line, (22, 283 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 130, 130), 1)

    cv2.putText(frame, "ESC to exit",
                (15, h_f - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1)


cap = cv2.VideoCapture(0)
print("Press ESC to exit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame     = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    ts_ms     = int((time.time() - start_time) * 1000)
    result    = detector.detect_for_video(mp_image, ts_ms)
    elapsed   = time.time() - start_time

    calibrating = len(green_signal) < WINDOW_SIZE
    progress    = len(green_signal) / WINDOW_SIZE

    if not result.face_landmarks:
        cv2.putText(frame, "NO FACE DETECTED", (30, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
        cv2.imshow("Face Spoof Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    landmarks = result.face_landmarks[0]
    h, w, _   = frame.shape
    curr_pts  = np.array([(lm.x, lm.y, lm.z) for lm in landmarks], dtype=np.float32)

    roi = extract_cheek_roi(frame, landmarks)
    if roi is not None:
        green_signal.append(float(np.mean(roi[:, :, 1])))

    if prev_landmarks is not None:
        rigidity_history.append(compute_rigidity(prev_landmarks, curr_pts))
    prev_landmarks = curr_pts

    left_ear  = eye_aspect_ratio(landmarks, LEFT_EYE_IDS,  w, h)
    right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE_IDS, w, h)
    ear       = (left_ear + right_ear) / 2.0

    if ear < BLINK_EAR_THRESH:
        consec_below += 1
    else:
        if consec_below >= BLINK_CONSEC:
            blink_total  += 1
        consec_below = 0

    avg_rigidity = float(np.mean(rigidity_history)) if rigidity_history else 0.0
    depth_var    = depth_variance(landmarks)

    face_crop = get_face_crop(frame, landmarks)
    lbp_score = lbp_uniformity(face_crop)

    pulse_energy = 0.0
    snr          = 0.0
    if not calibrating:
        filtered     = bandpass_filter(np.array(green_signal), FPS)
        pulse_energy = float(np.std(filtered))
        snr          = pulse_snr(filtered, FPS)

    scores = compute_scores(
        pulse_energy, snr, avg_rigidity, depth_var,
        blink_total, elapsed, lbp_score, calibrating
    )

    confidence = weighted_verdict(scores)

    if calibrating:
        verdict, color = "CALIBRATING", (200, 180, 0)
    elif confidence >= 0.62:
        verdict, color = f"REAL FACE  {confidence:.0%}", (0, 220, 0)
    elif confidence >= 0.45:
        verdict, color = f"UNCERTAIN  {confidence:.0%}", (0, 200, 200)
    else:
        low_scores = [k for k, v in scores.items() if v < 0.35]
        reason     = low_scores[0].upper() if low_scores else "MULTI"
        verdict    = f"SPOOF [{reason}]  {confidence:.0%}"
        color      = (0, 0, 255)

    metrics = {
        'pulse_e':  pulse_energy,
        'snr':      snr,
        'rigidity': avg_rigidity,
        'depth_v':  depth_var,
        'blinks':   blink_total,
    }

    draw_overlay(frame, verdict, color, scores, metrics, calibrating, progress)
    cv2.imshow("Face Spoof Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()