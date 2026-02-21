# Anti-Spoofing Pipeline — Architecture

_Last updated: 2026-02-21 12:44:04_

## Phase 1: Data Extraction and Preprocessing

**Input:** raw videos + selfie images  
**Output:** cropped frames ready for CNN  

### Steps
1. Walk the dataset directory tree for video and image files.
2. For videos — sample frames at 1–5 fps.
3. Detect faces using MTCNN (facenet-pytorch).
4. Crop and resize each detected face to 224×224 pixels.
5. Save crops organised into `data/frames/{dataset}/{real,spoof}/`.
6. Generate a CSV mapping every crop to its label and source.

### Folder layout
```
data/
  frames/
    <dataset_name>/
      real/
        frame_00001.jpg
        ...
      spoof/
        frame_00001.jpg
        ...
      labels.csv
```

---

## Phase 2: Dataset and DataLoader

**Script:** `antispoof_dataset.py`  
**Input:** face crops + `labels.csv` from Phase 1  
**Output:** PyTorch tensors ready for training  

### Features
1. **Single-frame mode** (`seq_len=1`) — returns one 224×224 image tensor.
2. **Sequence mode** (`seq_len > 1`) — returns N consecutive frames from the same source video as a `(N, C, H, W)` tensor (for temporal models).
3. Data augmentation (random flip, colour jitter, Gaussian blur) for training; deterministic resize + normalise for evaluation.
4. Label encoding: `real → 0`, `spoof → 1`.
5. Per-class sample counts and inverse-frequency class weights for imbalanced data.

---

## Phase 3: Baseline CNN Training

**Script:** `train_antispoof.py`  
**Input:** single-frame crops via `AntiSpoofDataset`  
**Output:** best model checkpoint (`checkpoints/best_antispoof.pt`)  

### Architecture
- **Backbone:** Pretrained ResNet-18 (ImageNet weights, fine-tuned).
- **Head:** `Linear(512, 1)` — single logit for binary classification.
- **Loss:** `BCEWithLogitsLoss` with `pos_weight` for class imbalance.
- **Optimiser:** AdamW (lr=1e-4, weight_decay=1e-4).
- **Scheduler (optional):** Cosine-annealing LR.

### Training loop
1. Load Phase 1 CSV → build train/val split (80/20 default).
2. Train for N epochs; log loss, accuracy, F1, precision, recall per epoch.
3. Save checkpoint when validation loss improves.
4. Final evaluation: reload best checkpoint → confusion matrix + metrics.

---

## Phase 4: Temporal Model

**Script:** `train_temporal.py`  
**Input:** sequence of frames per video via `AntiSpoofDataset` (seq_len > 1)  
**Output:** best temporal checkpoint (`checkpoints/best_temporal.pt`)  

### Architecture
- **CNN backbone:** ResNet-18 or MobileNetV3-Small (pretrained, optionally frozen).
- **Temporal aggregator** (collapses T frames → one descriptor):
  - **Average pooling** — simple mean across frames.
  - **Max pooling** — picks strongest activation per feature.
  - **Bidirectional LSTM** — learns order-dependent temporal patterns (e.g. screen flicker, moiré).
- **Classifier head:** `Dropout → Linear(D, 1)` → binary logit.
- **Loss:** `BCEWithLogitsLoss` with `pos_weight`.

### Data flow
```
(B, T, C, H, W)  →  reshape (B*T, C, H, W)
                  →  CNN backbone → (B*T, D)
                  →  reshape (B, T, D)
                  →  Temporal aggregator → (B, D')
                  →  Classifier → (B, 1) logit
```

### Why temporal?
Replay / screen attacks exhibit subtle temporal artefacts (flicker, unnatural
motion, moiré patterns) that single-frame models miss. Aggregating across
frames lets the model exploit these cues.