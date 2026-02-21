# Anti-Spoofing Pipeline — Architecture

_Last updated: 2026-02-21 11:28:57_

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
