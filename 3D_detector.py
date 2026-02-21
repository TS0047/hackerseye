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
RPPG_THRESHOLD   = 0.10
FREQ_SNR_MIN     = 1.8
RIGIDITY_LOWER   = 0.00005
RIGIDITY_UPPER   = 0.15
RIGIDITY_HISTORY = 45
DEPTH_VAR_MIN    = 0.002
BLINK_EAR_THRESH = 0.22
BLINK_CONSEC     = 2
MODEL_PATH       = 'face_landmarker.task'
CHEEK_IDS        = [234, 454, 93, 323]
LEFT_EYE_IDS     = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDS    = [33,  160, 158, 133, 153, 144]

WEIGHTS = {
    'pulse':   0.30,
    'motion':  0.20,
    'depth':   0.20,
    'blink':   0.15,
    'texture': 0.15,
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

green_signal      = deque(maxlen=WINDOW_SIZE)
rigidity_history  = deque(maxlen=RIGIDITY_HISTORY)
texture_baseline  = deque(maxlen=60)
blink_total       = 0
consec_below      = 0
prev_landmarks    = None
start_time        = time.time()


def bandpass_filter(signal, fs=30):
    nyq  = 0.5 * fs
    b, a = butter(3, [0.7 / nyq, 4.0 / nyq], btype='band')
    return filtfilt(b, a, signal)


def pulse_snr(signal, fs=30):
    freqs, psd = welch(signal, fs=fs, nperseg=min(len(signal), 256))
    mask       = (freqs >= 0.7) & (freqs <= 4.0)
    band_psd   = psd[mask]
    if len(band_psd) < 2 or band_psd.sum() == 0:
        return 0.0
    peak       = band_psd.max()
    noise      = (band_psd.sum() - peak) / max(len(band_psd) - 1, 1)
    return float(peak / (noise + 1e-8))


def extract_cheek_roi(frame, landmarks):
    h, w, _ = frame.shape
    xs  = [int(landmarks[i].x * w) for i in CHEEK_IDS]
    ys  = [int(landmarks[i].y * h) for i in CHEEK_IDS]
    roi = frame[min(ys):max(ys), min(xs):max(xs)]
    return roi if roi.size > 0 else None


def compute_rigidity(prev_pts, curr_pts):
    return float(np.std(np.linalg.norm(curr_pts - prev_pts, axis=1)))


def eye_aspect_ratio(landmarks, eye_ids, w, h):
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in eye_ids])
    A   = np.linalg.norm(pts[1] - pts[5])
    B   = np.linalg.norm(pts[2] - pts[4])
    C   = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C + 1e-8)


def depth_variance(landmarks):
    z = np.array([lm.z for lm in landmarks])
    return float(np.var(z))


def lbp_score_raw(roi):
    if roi is None or roi.size == 0:
        return None
    gray   = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi.copy()
    gray   = cv2.resize(gray, (96, 96)).astype(np.int16)
    padded = np.pad(gray, 1, mode='reflect')
    center = padded[1:-1, 1:-1]
    offsets = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
    lbp     = np.zeros((96, 96), dtype=np.uint8)
    for i, (dy, dx) in enumerate(offsets):
        nb   = padded[1+dy:97+dy, 1+dx:97+dx]
        lbp |= ((nb >= center).astype(np.uint8) << i)
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256), density=True)
    entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0] + 1e-10))
    return float(entropy)


def get_face_crop(frame, landmarks):
    h, w, _ = frame.shape
    xs  = [int(lm.x * w) for lm in landmarks]
    ys  = [int(lm.y * h) for lm in landmarks]
    pad = 20
    x1, x2 = max(min(xs) - pad, 0), min(max(xs) + pad, w)
    y1, y2 = max(min(ys) - pad, 0), min(max(ys) + pad, h)
    crop    = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def sigmoid(x, center, scale):
    return 1.0 / (1.0 + np.exp(-scale * (x - center)))


def compute_scores(pulse_energy, snr, avg_rigidity, depth_var,
                   blink_total, elapsed, lbp_raw, calibrating):
    if calibrating:
        return {k: 0.5 for k in WEIGHTS}

    scores = {}

    pulse_norm        = min(pulse_energy / 0.5, 1.0)
    snr_norm          = min(snr / 6.0, 1.0)
    scores['pulse']   = float(0.6 * pulse_norm + 0.4 * snr_norm)

    if avg_rigidity <= RIGIDITY_LOWER:
        scores['motion'] = 0.05
    elif avg_rigidity >= RIGIDITY_UPPER:
        scores['motion'] = float(sigmoid(avg_rigidity, RIGIDITY_UPPER, -60))
    else:
        ideal            = (RIGIDITY_LOWER + RIGIDITY_UPPER) / 2.0
        dist             = abs(avg_rigidity - ideal) / (ideal - RIGIDITY_LOWER)
        scores['motion'] = float(max(0.5, 1.0 - 0.5 * dist))

    scores['depth']   = float(min(depth_var / (DEPTH_VAR_MIN * 2.0), 1.0))

    if elapsed < 15:
        scores['blink'] = 0.5
    else:
        expected        = (elapsed / 60.0) * 15.0
        ratio           = blink_total / (expected + 1e-8)
        scores['blink'] = float(min(sigmoid(ratio, 0.4, 8.0), 1.0))

    if lbp_raw is None:
        scores['texture'] = 0.5
    else:
        if len(texture_baseline) >= 10:
            baseline          = float(np.median(texture_baseline))
            delta             = lbp_raw - baseline
            scores['texture'] = float(min(max(0.5 + delta * 2.0, 0.0), 1.0))
        else:
            scores['texture'] = 0.5

    return scores


def weighted_verdict(scores):
    return sum(WEIGHTS[k] * scores[k] for k in WEIGHTS)


def draw_overlay(frame, verdict, color, scores, metrics, calibrating, progress):
    h_f       = frame.shape[0]
    overlay   = frame.copy()
    cv2.rectangle(overlay, (15, 15), (450, 320), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    cv2.rectangle(frame, (15, 15), (450, 320), color, 2)

    if calibrating:
        bar_w = int(415 * progress)
        cv2.rectangle(frame, (20, 22), (435, 55), (40, 40, 40), -1)
        cv2.rectangle(frame, (20, 22), (20 + bar_w, 55), (170, 145, 0), -1)
        cv2.putText(frame, f"Calibrating...  {int(progress * 100)}%",
                    (28, 44), cv2.FONT_HERSHEY_DUPLEX, 0.62, (255, 255, 255), 1)
    else:
        cv2.putText(frame, verdict, (25, 55),
                    cv2.FONT_HERSHEY_DUPLEX, 0.95, color, 2)

    rows = [
        ("Pulse / rPPG",  scores.get('pulse',   0.5)),
        ("Motion",        scores.get('motion',  0.5)),
        ("Depth (3D)",    scores.get('depth',   0.5)),
        ("Blink rate",    scores.get('blink',   0.5)),
        ("Skin texture",  scores.get('texture', 0.5)),
    ]
    for i, (label, val) in enumerate(rows):
        y         = 90 + i * 36
        bar_full  = 175
        bar_fill  = int(bar_full * np.clip(val, 0, 1))
        g         = int(210 * val)
        r         = int(210 * (1 - val))
        cv2.rectangle(frame, (220, y - 14), (220 + bar_full, y + 6), (50, 50, 50), -1)
        cv2.rectangle(frame, (220, y - 14), (220 + bar_fill,  y + 6), (0, g, r), -1)
        cv2.putText(frame, label, (25, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (185, 185, 185), 1)
        cv2.putText(frame, f"{val:.2f}", (402, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (215, 215, 215), 1)

    cv2.line(frame, (15, 278), (450, 278), (55, 55, 55), 1)

    raw_lines = [
        f"rPPG: {metrics['pulse_e']:.3f} (thr {RPPG_THRESHOLD})   SNR: {metrics['snr']:.2f} (thr {FREQ_SNR_MIN})",
        f"Rigidity: {metrics['rigidity']:.5f} ({RIGIDITY_LOWER}-{RIGIDITY_UPPER})   "
        f"Depth: {metrics['depth_v']:.5f}   Blinks: {metrics['blinks']}",
        f"LBP raw: {metrics['lbp']:.3f}   LBP baseline: {metrics['lbp_base']:.3f}",
    ]
    for i, line in enumerate(raw_lines):
        cv2.putText(frame, line, (22, 294 + i * 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (125, 125, 125), 1)

    cv2.putText(frame, "ESC to exit",
                (15, h_f - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (90, 90, 90), 1)


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

    ear = (eye_aspect_ratio(landmarks, LEFT_EYE_IDS,  w, h) +
           eye_aspect_ratio(landmarks, RIGHT_EYE_IDS, w, h)) / 2.0
    if ear < BLINK_EAR_THRESH:
        consec_below += 1
    else:
        if consec_below >= BLINK_CONSEC:
            blink_total += 1
        consec_below = 0

    avg_rigidity = float(np.mean(rigidity_history)) if rigidity_history else 0.0
    depth_var    = depth_variance(landmarks)
    face_crop    = get_face_crop(frame, landmarks)
    lbp_raw      = lbp_score_raw(face_crop)

    if lbp_raw is not None:
        texture_baseline.append(lbp_raw)

    lbp_base = float(np.median(texture_baseline)) if texture_baseline else 0.0

    pulse_energy = 0.0
    snr          = 0.0
    if not calibrating:
        filtered     = bandpass_filter(np.array(green_signal), FPS)
        pulse_energy = float(np.std(filtered))
        snr          = pulse_snr(filtered, FPS)

    scores     = compute_scores(pulse_energy, snr, avg_rigidity, depth_var,
                                blink_total, elapsed, lbp_raw, calibrating)
    confidence = weighted_verdict(scores)

    if calibrating:
        verdict, color = "CALIBRATING", (200, 175, 0)
    elif confidence >= 0.60:
        verdict = f"REAL FACE  {confidence:.0%}"
        color   = (0, 215, 0)
    elif confidence >= 0.44:
        verdict = f"UNCERTAIN  {confidence:.0%}"
        color   = (0, 195, 195)
    else:
        low  = sorted(scores, key=scores.get)
        why  = low[0].upper()
        verdict = f"SPOOF [{why}]  {confidence:.0%}"
        color   = (0, 0, 245)

    metrics = {
        'pulse_e':  pulse_energy,
        'snr':      snr,
        'rigidity': avg_rigidity,
        'depth_v':  depth_var,
        'blinks':   blink_total,
        'lbp':      lbp_raw if lbp_raw is not None else 0.0,
        'lbp_base': lbp_base,
    }

    draw_overlay(frame, verdict, color, scores, metrics, calibrating, progress)
    cv2.imshow("Face Spoof Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()