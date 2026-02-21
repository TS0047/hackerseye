import cv2
import numpy as np
import mediapipe as mp
import time
import os
import json
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import mobilenet_v2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.signal import butter, filtfilt

FPS               = 30
LANDMARK_SEQ_LEN  = 60
RPPG_WINDOW       = 180
FACE_CROP_SIZE    = 128
MODEL_PATH        = 'face_landmarker.task'
WEIGHTS_DIR       = 'spoof_weights'
TRAINING_DATA_DIR = 'spoof_training_data'
LSTM_INPUT_DIM    = 478 * 3
LSTM_HIDDEN       = 256
LSTM_LAYERS       = 2
CNN_FEATURE_DIM   = 128
SPOOF_THRESHOLD   = 0.5
ADAPT_LR          = 1e-4
FINETUNE_EPOCHS   = 5
DEVICE            = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHEEK_IDS         = [234, 454, 93, 323]

os.makedirs(WEIGHTS_DIR, 0o755, exist_ok=True)
os.makedirs(TRAINING_DATA_DIR, 0o755, exist_ok=True)

FUSION_WEIGHTS_PATH      = os.path.join(WEIGHTS_DIR, 'fusion.pth')
LSTM_WEIGHTS_PATH        = os.path.join(WEIGHTS_DIR, 'lstm.pth')
RPPG_WEIGHTS_PATH        = os.path.join(WEIGHTS_DIR, 'rppg.pth')
CNN_WEIGHTS_PATH         = os.path.join(WEIGHTS_DIR, 'cnn.pth')
TRAINING_LOG_PATH        = os.path.join(TRAINING_DATA_DIR, 'training_log.json')
SAMPLE_STORE_PATH        = os.path.join(TRAINING_DATA_DIR, 'samples.npz')
SESSION_STATS_PATH       = os.path.join(TRAINING_DATA_DIR, 'session_stats.json')


class LandmarkLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(LSTM_INPUT_DIM)
        self.lstm = nn.LSTM(
            input_size=LSTM_INPUT_DIM,
            hidden_size=LSTM_HIDDEN,
            num_layers=LSTM_LAYERS,
            batch_first=True,
            dropout=0.3
        )
        self.attention = nn.Sequential(
            nn.Linear(LSTM_HIDDEN, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(LSTM_HIDDEN, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.norm(x)
        out, _ = self.lstm(x)
        attn = F.softmax(self.attention(out), dim=1)
        ctx = (attn * out).sum(dim=1)
        return torch.sigmoid(self.classifier(ctx))


class FaceCNN(nn.Module):
    def __init__(self):
        super().__init__()
        base = mobilenet_v2(weights=None)
        base.classifier = nn.Identity()
        self.backbone = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, CNN_FEATURE_DIM),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


class RPPGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = self.net(x).squeeze(-1)
        return torch.sigmoid(self.classifier(x))


class FusionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, lstm_s, cnn_s, rppg_s):
        x = torch.cat([lstm_s, cnn_s, rppg_s], dim=1)
        return torch.sigmoid(self.net(x))


def save_weights(lstm_model, cnn_model, rppg_model, fusion_model):
    torch.save(lstm_model.state_dict(),   LSTM_WEIGHTS_PATH)
    torch.save(cnn_model.state_dict(),    CNN_WEIGHTS_PATH)
    torch.save(rppg_model.state_dict(),   RPPG_WEIGHTS_PATH)
    torch.save(fusion_model.state_dict(), FUSION_WEIGHTS_PATH)
    print("[Weights] Saved all model weights.")


def load_weights(lstm_model, cnn_model, rppg_model, fusion_model):
    loaded = []
    pairs = [
        (LSTM_WEIGHTS_PATH,    lstm_model,   "LSTM"),
        (CNN_WEIGHTS_PATH,     cnn_model,    "CNN"),
        (RPPG_WEIGHTS_PATH,    rppg_model,   "RPPGNet"),
        (FUSION_WEIGHTS_PATH,  fusion_model, "Fusion"),
    ]
    for path, model, name in pairs:
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            loaded.append(name)
    if loaded:
        print(f"[Weights] Loaded: {', '.join(loaded)}")
    else:
        print("[Weights] No saved weights found. Starting fresh.")


def load_training_log():
    if os.path.exists(TRAINING_LOG_PATH):
        with open(TRAINING_LOG_PATH, 'r') as f:
            return json.load(f)
    return {"entries": [], "real_count": 0, "spoof_count": 0, "total_sessions": 0}


def save_training_log(log):
    with open(TRAINING_LOG_PATH, 'w') as f:
        json.dump(log, f, indent=2)


def load_sample_store():
    if os.path.exists(SAMPLE_STORE_PATH):
        data = np.load(SAMPLE_STORE_PATH, allow_pickle=True)
        features = list(data['features'])
        labels   = list(data['labels'])
        print(f"[Samples] Loaded {len(features)} stored training samples.")
        return features, labels
    return [], []


def save_sample_store(features, labels):
    np.savez_compressed(
        SAMPLE_STORE_PATH,
        features=np.array(features, dtype=object),
        labels=np.array(labels, dtype=np.float32)
    )
    print(f"[Samples] Saved {len(features)} training samples.")


def load_session_stats():
    if os.path.exists(SESSION_STATS_PATH):
        with open(SESSION_STATS_PATH, 'r') as f:
            return json.load(f)
    return {"total_sessions": 0, "total_labels": 0, "real_labels": 0, "spoof_labels": 0}


def save_session_stats(stats):
    with open(SESSION_STATS_PATH, 'w') as f:
        json.dump(stats, f, indent=2)


def retrain_on_stored_samples(fusion_model, stored_features, stored_labels, epochs=FINETUNE_EPOCHS):
    if len(stored_features) < 2:
        return
    print(f"[Retrain] Fine-tuning on {len(stored_features)} stored samples for {epochs} epochs...")
    opt = torch.optim.Adam(fusion_model.parameters(), lr=ADAPT_LR)
    loss_fn = nn.BCELoss()
    fusion_model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        indices = np.random.permutation(len(stored_features))
        for idx in indices:
            feat  = stored_features[idx]
            label = torch.tensor([[stored_labels[idx]]], dtype=torch.float32, device=DEVICE)
            l = torch.tensor([[feat[0]]], dtype=torch.float32, device=DEVICE)
            c = torch.tensor([[feat[1]]], dtype=torch.float32, device=DEVICE)
            r = torch.tensor([[feat[2]]], dtype=torch.float32, device=DEVICE)
            pred = fusion_model(l, c, r)
            loss = loss_fn(pred, label)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}/{epochs}  loss: {total_loss/len(stored_features):.4f}")
    fusion_model.eval()
    print("[Retrain] Done.")


def bandpass_filter(signal, fs=30):
    nyq  = 0.5 * fs
    low  = 0.7 / nyq
    high = 4.0 / nyq
    b, a = butter(3, [low, high], btype='band')
    return filtfilt(b, a, signal)


def extract_cheek_roi(frame, landmarks):
    h, w, _ = frame.shape
    xs = [int(landmarks[i].x * w) for i in CHEEK_IDS]
    ys = [int(landmarks[i].y * h) for i in CHEEK_IDS]
    roi = frame[min(ys):max(ys), min(xs):max(xs)]
    return roi if roi.size > 0 else None


def normalize_landmarks(landmarks):
    pts = np.array([(lm.x, lm.y, lm.z) for lm in landmarks], dtype=np.float32)
    pts -= pts.mean(axis=0)
    scale = np.linalg.norm(pts, axis=1).max() + 1e-8
    pts /= scale
    return pts.flatten()


def preprocess_face(frame, landmarks):
    h, w, _ = frame.shape
    xs  = [int(lm.x * w) for lm in landmarks]
    ys  = [int(lm.y * h) for lm in landmarks]
    pad = 30
    x1  = max(min(xs) - pad, 0)
    x2  = min(max(xs) + pad, w)
    y1  = max(min(ys) - pad, 0)
    y2  = min(max(ys) + pad, h)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop   = cv2.resize(crop, (FACE_CROP_SIZE, FACE_CROP_SIZE))
    crop   = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])(crop)
    return tensor.unsqueeze(0).to(DEVICE)


def draw_panel(frame, verdict, color, scores, calibrating, progress, stats):
    overlay = frame.copy()
    cv2.rectangle(overlay, (15, 15), (440, 270), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.rectangle(frame, (15, 15), (440, 270), color, 2)

    if calibrating:
        bar_w = int(405 * progress)
        cv2.rectangle(frame, (15, 15), (420, 55), (40, 40, 40), -1)
        cv2.rectangle(frame, (15, 15), (15 + bar_w, 55), (180, 160, 0), -1)
        cv2.putText(frame, f"Calibrating  {int(progress * 100)}%", (25, 42),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1)
    else:
        cv2.putText(frame, verdict, (25, 55),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)

    rows = [
        ("LSTM  (motion pattern)", scores.get('lstm',   0.0)),
        ("CNN   (texture/depth)",  scores.get('cnn',    0.0)),
        ("rPPG  (pulse signal)",   scores.get('rppg',   0.0)),
        ("FUSION (final)",         scores.get('fusion', 0.0)),
    ]
    for i, (label, score) in enumerate(rows):
        y         = 90 + i * 36
        bar_full  = 190
        bar_fill  = int(bar_full * score)
        bar_color = (0, int(210 * score), int(210 * (1 - score)))
        cv2.rectangle(frame, (210, y - 14), (210 + bar_full, y + 5), (50, 50, 50), -1)
        cv2.rectangle(frame, (210, y - 14), (210 + bar_fill,  y + 5), bar_color, -1)
        cv2.putText(frame, label, (25, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (190, 190, 190), 1)
        cv2.putText(frame, f"{score:.3f}", (408, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1)

    cv2.line(frame, (15, 238), (440, 238), (60, 60, 60), 1)
    cv2.putText(frame,
                f"Session labels: R={stats['session_real']}  S={stats['session_spoof']}  "
                f"| All-time: R={stats['total_real']}  S={stats['total_spoof']}",
                (22, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 150, 150), 1)

    cv2.putText(frame, "R=mark real  S=mark spoof  W=save  Q/ESC=quit+save",
                (15, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 120, 120), 1)


def main():
    print(f"[Init] Using device: {DEVICE}")

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    mp_options   = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector = vision.FaceLandmarker.create_from_options(mp_options)

    lstm_model   = LandmarkLSTM().to(DEVICE).eval()
    cnn_model    = FaceCNN().to(DEVICE).eval()
    rppg_model   = RPPGNet().to(DEVICE).eval()
    fusion_model = FusionHead().to(DEVICE).eval()

    load_weights(lstm_model, cnn_model, rppg_model, fusion_model)

    stored_features, stored_labels = load_sample_store()
    training_log  = load_training_log()
    session_stats = load_session_stats()

    if len(stored_features) >= 2:
        retrain_on_stored_samples(fusion_model, stored_features, stored_labels)

    adapt_opt    = torch.optim.Adam(fusion_model.parameters(), lr=ADAPT_LR)
    loss_fn      = nn.BCELoss()
    landmark_seq = collections.deque(maxlen=LANDMARK_SEQ_LEN)
    rppg_signal  = collections.deque(maxlen=RPPG_WINDOW)

    scores             = {'lstm': 0.0, 'cnn': 0.0, 'rppg': 0.0, 'fusion': 0.0}
    verdict            = "CALIBRATING"
    verdict_color      = (200, 180, 0)
    last_fusion_pred   = None
    last_feature_vec   = None
    start_time         = time.time()

    session_real  = 0
    session_spoof = 0

    cap = cv2.VideoCapture(0)
    print("[Ready] Press R=real  S=spoof  W=save now  Q/ESC=quit+save")

    def do_save():
        save_weights(lstm_model, cnn_model, rppg_model, fusion_model)
        save_sample_store(stored_features, stored_labels)
        training_log['total_sessions'] = session_stats['total_sessions']
        save_training_log(training_log)
        session_stats['total_sessions'] += 1
        session_stats['total_labels']   += session_real + session_spoof
        session_stats['real_labels']    += session_real
        session_stats['spoof_labels']   += session_spoof
        save_session_stats(session_stats)
        print(f"[Save] Session: real={session_real}  spoof={session_spoof}  "
              f"total_samples={len(stored_features)}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame     = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        ts_ms     = int((time.time() - start_time) * 1000)
        result    = detector.detect_for_video(mp_image, ts_ms)

        calibrating = (len(landmark_seq) < LANDMARK_SEQ_LEN or
                       len(rppg_signal)  < RPPG_WINDOW)
        progress    = min(
            len(landmark_seq) / LANDMARK_SEQ_LEN,
            len(rppg_signal)  / RPPG_WINDOW
        )

        display_stats = {
            'session_real':  session_real,
            'session_spoof': session_spoof,
            'total_real':    session_stats.get('real_labels',  0),
            'total_spoof':   session_stats.get('spoof_labels', 0),
        }

        if not result.face_landmarks:
            cv2.putText(frame, "NO FACE DETECTED", (30, 50),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
            draw_panel(frame, verdict, verdict_color, scores,
                       calibrating, progress, display_stats)
            cv2.imshow("Face Spoof Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                do_save()
                break
            continue

        landmarks = result.face_landmarks[0]
        norm_pts  = normalize_landmarks(landmarks)
        landmark_seq.append(norm_pts)

        roi = extract_cheek_roi(frame, landmarks)
        if roi is not None:
            rppg_signal.append(float(np.mean(roi[:, :, 1])))

        with torch.no_grad():
            if len(landmark_seq) == LANDMARK_SEQ_LEN:
                seq_t      = torch.tensor(np.array(landmark_seq),
                                          dtype=torch.float32).unsqueeze(0).to(DEVICE)
                lstm_score = lstm_model(seq_t).item()
                scores['lstm'] = lstm_score
            else:
                lstm_score = scores['lstm']

            face_t = preprocess_face(frame, landmarks)
            if face_t is not None:
                feat      = cnn_model(face_t)
                cnn_score = torch.sigmoid(feat.mean()).item()
                scores['cnn'] = cnn_score
            else:
                cnn_score = scores['cnn']

            if len(rppg_signal) == RPPG_WINDOW:
                sig = np.array(rppg_signal, dtype=np.float32)
                sig = bandpass_filter(sig, FPS)
                sig = (sig - sig.mean()) / (sig.std() + 1e-8)
                rppg_t     = torch.tensor(sig, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
                rppg_score = rppg_model(rppg_t).item()
                scores['rppg'] = rppg_score
            else:
                rppg_score = scores['rppg']

            if not calibrating:
                l = torch.tensor([[lstm_score]], dtype=torch.float32, device=DEVICE)
                c = torch.tensor([[cnn_score]],  dtype=torch.float32, device=DEVICE)
                r = torch.tensor([[rppg_score]], dtype=torch.float32, device=DEVICE)
                fusion_pred  = fusion_model(l, c, r)
                last_fusion_pred  = fusion_pred
                last_feature_vec  = [lstm_score, cnn_score, rppg_score]
                fusion_score      = fusion_pred.item()
                scores['fusion']  = fusion_score

                if fusion_score >= SPOOF_THRESHOLD:
                    verdict, verdict_color = "REAL FACE",      (0, 220, 0)
                else:
                    verdict, verdict_color = "SPOOF DETECTED", (0, 0, 255)

        draw_panel(frame, verdict, verdict_color, scores,
                   calibrating, progress, display_stats)
        cv2.imshow("Face Spoof Detection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord('q')):
            do_save()
            break

        elif key == ord('w'):
            do_save()

        elif key == ord('r') and last_fusion_pred is not None and last_feature_vec is not None:
            label_t = torch.tensor([[1.0]], dtype=torch.float32, device=DEVICE)
            loss    = loss_fn(last_fusion_pred, label_t)
            adapt_opt.zero_grad()
            loss.backward()
            adapt_opt.step()
            stored_features.append(last_feature_vec)
            stored_labels.append(1.0)
            session_real += 1
            entry = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "label": "real",
                "features": last_feature_vec,
                "loss": loss.item()
            }
            training_log['entries'].append(entry)
            training_log['real_count'] += 1
            print(f"[Label] REAL   | loss={loss.item():.4f}  "
                  f"total_samples={len(stored_features)}")

        elif key == ord('s') and last_fusion_pred is not None and last_feature_vec is not None:
            label_t = torch.tensor([[0.0]], dtype=torch.float32, device=DEVICE)
            loss    = loss_fn(last_fusion_pred, label_t)
            adapt_opt.zero_grad()
            loss.backward()
            adapt_opt.step()
            stored_features.append(last_feature_vec)
            stored_labels.append(0.0)
            session_spoof += 1
            entry = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "label": "spoof",
                "features": last_feature_vec,
                "loss": loss.item()
            }
            training_log['entries'].append(entry)
            training_log['spoof_count'] += 1
            print(f"[Label] SPOOF  | loss={loss.item():.4f}  "
                  f"total_samples={len(stored_features)}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()