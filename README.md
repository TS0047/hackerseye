# 🎥 Live Face Recognition System

Real-time face recognition using **InsightFace (ArcFace)** + **SQLite** + **OpenCV**.

```
Webcam Frame → Detect Face → 512D Embedding → Compare DB → Draw Box + Name
```

---

## 📁 Files

| File | Purpose |
|------|---------|
| `database.py` | SQLite helpers — init, add, get, delete |
| `register.py` | Capture + register a new person |
| `live_recognition.py` | Real-time webcam recognition |
| `manage_db.py` | CLI to list / delete users |
| `requirements.txt` | Python dependencies |

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

> **No NVIDIA GPU?** Replace `onnxruntime-gpu` with `onnxruntime` in requirements.txt
> and set `CTX_ID = -1` in `live_recognition.py` and `register.py`.

InsightFace will auto-download the `buffalo_l` model (~300 MB) on first run.

---

## 🪜 Step 1 — Register People

```bash
python register.py --name "Alice"
python register.py --name "Bob" --samples 8
python register_from_video.py --name "Alice" --video alice.mp4

python register_from_video.py --name "Alice" --webcam --duration 5
```

- A webcam window opens.
- Press **SPACE** to capture each sample (default: 5 samples per person).
- Embeddings are averaged for better accuracy.
- Press **Q** to cancel.

---

## 🪜 Step 2 — Run Live Recognition

```bash
python live_recognition.py
```

| Key | Action |
|-----|--------|
| **Q** | Quit |
| **R** | Reload database (add new users without restarting) |
| **S** | Toggle FPS / stats overlay |

### What you'll see
- 🟢 **Green box** → Recognised person + name + similarity score
- 🔴 **Red box** → Unknown face

---

## 🛠 Manage the Database

```bash
python manage_db.py list               # List all users
python manage_db.py delete --name Bob  # Delete one person
python manage_db.py reset              # Wipe everything
```

---

## 🎯 Tuning

| Parameter | Location | Effect |
|-----------|----------|--------|
| `THRESHOLD` | `live_recognition.py` | Lower = stricter matching (fewer false positives) |
| `--samples` | `register.py` | More samples = more robust embedding |
| `CTX_ID` | both scripts | `0` = GPU, `-1` = CPU |

A threshold of **0.45** works well in normal lighting. Try **0.40–0.50** range.

---

## 🧱 Architecture

```
buffalo_l model (InsightFace)
├── RetinaFace — face detection + landmarks
└── ArcFace    — 512-dimensional face embedding

SQLite (faces.db)
└── users table: id | name | embedding (blob) | created_at

Matching
└── Cosine similarity between live embedding and all DB embeddings
    → Pick highest score; label Unknown if below threshold
```
