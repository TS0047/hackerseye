"""
extract_faces.py — Phase 1: Face crop extraction for anti-spoofing pipeline.

On every run this script will:
  1.  Write / update  architecture.md  with the current phase description.
  2.  Walk the dataset tree looking for videos (.mp4, .avi, .mov, .mkv)
      and images (.jpg, .jpeg, .png, .bmp).
  3.  For videos: sample frames at a configurable FPS (default 3 fps),
      detect faces with MTCNN, crop & resize to 224×224, and save as JPG.
  4.  For images: detect and crop faces the same way.
  5.  Organise output into:
          data/frames/{dataset_name}/{real,spoof}/
  6.  Generate  data/frames/{dataset_name}/labels.csv  mapping each
      output image → label & source file.
  7.  Log every processed file; skip files that fail detection.

Usage:
    python extract_faces.py                        # use .env defaults
    python extract_faces.py --dataset_root <path>  # override root
    python extract_faces.py --fps 5                # override sample rate

Environment variables (loaded from .env):
    DATASET_ROOT      — path to the raw dataset
    OUTPUT_DIR        — base output directory  (default: data/frames)
    FACE_CROP_SIZE    — target crop size        (default: 224)
    VIDEO_SAMPLE_FPS  — frames per second       (default: 3)
    LOG_LEVEL         — logging verbosity        (default: INFO)
"""

# ═══════════════════════════════════════════════════════════════
# Imports
# ═══════════════════════════════════════════════════════════════
import os
import sys
import csv
import cv2
import logging
import argparse
import datetime
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

# MTCNN from facenet-pytorch for robust face detection
from facenet_pytorch import MTCNN

# python-dotenv to read .env config
from dotenv import load_dotenv

# ═══════════════════════════════════════════════════════════════
# Load environment variables from .env (if present)
# ═══════════════════════════════════════════════════════════════
load_dotenv()

# ═══════════════════════════════════════════════════════════════
# Constants & defaults (overridable via .env or CLI)
# ═══════════════════════════════════════════════════════════════
DEFAULT_DATASET_ROOT  = os.getenv("DATASET_ROOT", "")
DEFAULT_OUTPUT_DIR    = os.getenv("OUTPUT_DIR", "data/frames")
DEFAULT_CROP_SIZE     = int(os.getenv("FACE_CROP_SIZE", "224"))
DEFAULT_SAMPLE_FPS    = int(os.getenv("VIDEO_SAMPLE_FPS", "3"))
DEFAULT_LOG_LEVEL     = os.getenv("LOG_LEVEL", "INFO")

# File extensions we handle
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ═══════════════════════════════════════════════════════════════
# 0.  Architecture doc writer
# ═══════════════════════════════════════════════════════════════
def write_architecture_doc(project_root: str) -> None:
    """
    Create or update 'architecture.md' at the project root with the
    current pipeline phase description.

    This is called once at the start of every run so the doc always
    reflects the latest state of the pipeline.
    """
    arch_path = os.path.join(project_root, "architecture.md")

    content = (
        "# Anti-Spoofing Pipeline — Architecture\n"
        "\n"
        f"_Last updated: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}_\n"
        "\n"
        "## Phase 1: Data Extraction and Preprocessing\n"
        "\n"
        "**Input:** raw videos + selfie images  \n"
        "**Output:** cropped frames ready for CNN  \n"
        "\n"
        "### Steps\n"
        "1. Walk the dataset directory tree for video and image files.\n"
        "2. For videos — sample frames at 1–5 fps.\n"
        "3. Detect faces using MTCNN (facenet-pytorch).\n"
        "4. Crop and resize each detected face to 224×224 pixels.\n"
        "5. Save crops organised into `data/frames/{dataset}/{real,spoof}/`.\n"
        "6. Generate a CSV mapping every crop to its label and source.\n"
        "\n"
        "### Folder layout\n"
        "```\n"
        "data/\n"
        "  frames/\n"
        "    <dataset_name>/\n"
        "      real/\n"
        "        frame_00001.jpg\n"
        "        ...\n"
        "      spoof/\n"
        "        frame_00001.jpg\n"
        "        ...\n"
        "      labels.csv\n"
        "```\n"
    )

    # Write (or overwrite) the architecture doc
    with open(arch_path, "w", encoding="utf-8") as fh:
        fh.write(content)

    logging.info("architecture.md written at %s", arch_path)


# ═══════════════════════════════════════════════════════════════
# 1.  Logger setup
# ═══════════════════════════════════════════════════════════════
def setup_logging(level_name: str = "INFO") -> logging.Logger:
    """
    Configure and return the root logger with a timestamped format.

    Parameters
    ----------
    level_name : str
        One of DEBUG, INFO, WARNING, ERROR, CRITICAL.

    Returns
    -------
    logging.Logger
        Configured root logger instance.
    """
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 2.  MTCNN detector initialisation
# ═══════════════════════════════════════════════════════════════
def create_detector(crop_size: int = 224) -> MTCNN:
    """
    Instantiate and return an MTCNN face detector.

    Parameters
    ----------
    crop_size : int
        The square dimension for output face crops (default 224).

    Returns
    -------
    MTCNN
        Ready-to-use detector that returns PIL images of cropped faces.
    """
    detector = MTCNN(
        image_size=crop_size,   # output face size
        margin=20,              # extra pixels around the bounding box
        keep_all=True,          # detect ALL faces in a frame
        post_process=False,     # skip standardisation — we want raw pixels
        device="cuda" if _cuda_available() else "cpu",
    )
    logging.info(
        "MTCNN initialised  (crop=%dpx, device=%s)",
        crop_size,
        "cuda" if _cuda_available() else "cpu",
    )
    return detector


def _cuda_available() -> bool:
    """Check whether PyTorch can see a CUDA GPU."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ═══════════════════════════════════════════════════════════════
# 3.  Label inference from path
# ═══════════════════════════════════════════════════════════════
def infer_label(filepath: str) -> str:
    """
    Infer the anti-spoofing label from the file's path components.

    The dataset is expected to contain folder names such as
    'real', 'live', 'genuine'  →  "real"
    'spoof', 'fake', 'print', 'replay', 'mask'  →  "spoof"

    If no known keyword is found the file is classified as "unknown".

    Parameters
    ----------
    filepath : str
        Full path to the source video or image.

    Returns
    -------
    str
        One of "real", "spoof", or "unknown".
    """
    # Normalise separators and lower-case for matching
    parts = filepath.replace("\\", "/").lower()

    real_keywords  = ["real", "live", "genuine", "bonafide", "bona_fide"]
    spoof_keywords = ["spoof", "fake", "print", "replay", "mask", "attack", "photo"]

    # Check every keyword against any directory component
    for kw in real_keywords:
        if kw in parts:
            return "real"
    for kw in spoof_keywords:
        if kw in parts:
            return "spoof"

    return "unknown"


# ═══════════════════════════════════════════════════════════════
# 4.  Face detection helpers
# ═══════════════════════════════════════════════════════════════
def detect_and_crop_faces(
    detector: MTCNN,
    rgb_image: np.ndarray,
    crop_size: int = 224,
) -> List[np.ndarray]:
    """
    Run MTCNN on an RGB numpy image and return a list of face crops.

    Each crop is a uint8 numpy array of shape (crop_size, crop_size, 3).

    Parameters
    ----------
    detector : MTCNN
        The MTCNN instance.
    rgb_image : np.ndarray
        HxWx3 RGB image.
    crop_size : int
        Target square size for each face crop.

    Returns
    -------
    List[np.ndarray]
        Possibly empty list of face crop arrays.
    """
    # Convert numpy to PIL (MTCNN expects PIL or tensor)
    pil_img = Image.fromarray(rgb_image)

    # Detect — returns tensor of shape (N, 3, H, W) or None
    faces_tensor = detector(pil_img)

    if faces_tensor is None:
        return []

    crops: List[np.ndarray] = []
    for face_t in faces_tensor:
        # face_t is (3, H, W) float tensor — convert to uint8 HWC numpy
        face_np = face_t.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

        # Ensure exact target size (should already be, but be safe)
        if face_np.shape[0] != crop_size or face_np.shape[1] != crop_size:
            face_np = cv2.resize(face_np, (crop_size, crop_size))

        crops.append(face_np)

    return crops


# ═══════════════════════════════════════════════════════════════
# 5.  Video frame extraction
# ═══════════════════════════════════════════════════════════════
def extract_faces_from_video(
    video_path: str,
    detector: MTCNN,
    output_dir: str,
    label: str,
    sample_fps: int = 3,
    crop_size: int = 224,
    counter_start: int = 0,
) -> Tuple[int, List[dict]]:
    """
    Sample frames from a video at the given FPS, detect faces with MTCNN,
    crop to the target size, and save as JPEG.

    Parameters
    ----------
    video_path : str
        Path to the source video file.
    detector : MTCNN
        Face detection model.
    output_dir : str
        Directory where crops will be saved (e.g. data/frames/ds/real/).
    label : str
        "real" or "spoof" — written into the CSV record.
    sample_fps : int
        How many frames per second to sample from the video (1-5).
    crop_size : int
        Target face crop size in pixels.
    counter_start : int
        Starting index for filenames to avoid overwrites across files.

    Returns
    -------
    Tuple[int, List[dict]]
        (number of crops saved, list of CSV row dicts).
    """
    logger = logging.getLogger(__name__)
    logger.info("Processing video: %s  (label=%s)", video_path, label)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("Cannot open video: %s — skipping", video_path)
        return 0, []

    # Determine actual FPS and compute the frame-skip interval
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # How many source frames to skip between samples
    skip = max(1, int(round(video_fps / sample_fps)))

    logger.debug(
        "  video_fps=%.1f  total_frames=%d  skip=%d",
        video_fps, total_frames, skip,
    )

    saved = 0
    records: List[dict] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # end of video

        # Only process every `skip`-th frame
        if frame_idx % skip != 0:
            frame_idx += 1
            continue
        frame_idx += 1

        # OpenCV reads BGR — convert to RGB for MTCNN
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            crops = detect_and_crop_faces(detector, rgb, crop_size)
        except Exception as exc:
            logger.debug("  Frame %d detection error: %s", frame_idx, exc)
            continue

        if not crops:
            logger.debug("  Frame %d — no faces detected", frame_idx)
            continue

        # Save each detected face crop
        for crop in crops:
            idx = counter_start + saved
            filename = f"frame_{idx:06d}.jpg"
            out_path = os.path.join(output_dir, filename)

            # Save as BGR JPEG (OpenCV expects BGR)
            bgr_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            cv2.imwrite(out_path, bgr_crop)

            records.append({
                "filename": filename,
                "label": label,
                "source": os.path.basename(video_path),
                "source_type": "video",
                "frame_idx": frame_idx,
            })
            saved += 1

    cap.release()
    logger.info("  → saved %d face crops from %s", saved, video_path)
    return saved, records


# ═══════════════════════════════════════════════════════════════
# 6.  Image face extraction
# ═══════════════════════════════════════════════════════════════
def extract_faces_from_image(
    image_path: str,
    detector: MTCNN,
    output_dir: str,
    label: str,
    crop_size: int = 224,
    counter_start: int = 0,
) -> Tuple[int, List[dict]]:
    """
    Detect and crop faces from a single image file.

    Parameters
    ----------
    image_path : str
        Path to the source image.
    detector : MTCNN
        MTCNN face detector.
    output_dir : str
        Target directory for saving crops.
    label : str
        "real" or "spoof".
    crop_size : int
        Target face crop size.
    counter_start : int
        Filename index offset.

    Returns
    -------
    Tuple[int, List[dict]]
        (number of crops saved, list of CSV row dicts).
    """
    logger = logging.getLogger(__name__)
    logger.info("Processing image: %s  (label=%s)", image_path, label)

    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.warning("Cannot read image: %s — skipping", image_path)
            return 0, []

        # Convert BGR → RGB for MTCNN
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        crops = detect_and_crop_faces(detector, rgb, crop_size)
    except Exception as exc:
        logger.warning("Error processing %s: %s — skipping", image_path, exc)
        return 0, []

    if not crops:
        logger.debug("  No faces detected in %s", image_path)
        return 0, []

    saved = 0
    records: List[dict] = []

    for crop in crops:
        idx = counter_start + saved
        filename = f"frame_{idx:06d}.jpg"
        out_path = os.path.join(output_dir, filename)

        bgr_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, bgr_crop)

        records.append({
            "filename": filename,
            "label": label,
            "source": os.path.basename(image_path),
            "source_type": "image",
            "frame_idx": 0,
        })
        saved += 1

    logger.info("  → saved %d face crops from %s", saved, image_path)
    return saved, records


# ═══════════════════════════════════════════════════════════════
# 7.  Dataset walker — discover all videos & images
# ═══════════════════════════════════════════════════════════════
def discover_files(root: str) -> Tuple[List[str], List[str]]:
    """
    Recursively walk `root` and collect paths to video and image files.

    Parameters
    ----------
    root : str
        Top-level dataset directory.

    Returns
    -------
    Tuple[List[str], List[str]]
        (list_of_video_paths, list_of_image_paths)
    """
    videos: List[str] = []
    images: List[str] = []

    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            full = os.path.join(dirpath, fname)
            if ext in VIDEO_EXTENSIONS:
                videos.append(full)
            elif ext in IMAGE_EXTENSIONS:
                images.append(full)

    logging.info(
        "Discovered %d videos and %d images under %s",
        len(videos), len(images), root,
    )
    return videos, images


# ═══════════════════════════════════════════════════════════════
# 8.  CSV writer
# ═══════════════════════════════════════════════════════════════
def write_csv(records: List[dict], csv_path: str) -> None:
    """
    Write the list of record dicts to a CSV file.

    Each row contains: filename, label, source, source_type, frame_idx.

    Parameters
    ----------
    records : List[dict]
        Rows to write.
    csv_path : str
        Output CSV file path.
    """
    if not records:
        logging.warning("No records to write — CSV skipped.")
        return

    fieldnames = ["filename", "label", "source", "source_type", "frame_idx"]

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    logging.info("CSV written: %s  (%d rows)", csv_path, len(records))


# ═══════════════════════════════════════════════════════════════
# 9.  CLI argument parser
# ═══════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    """
    Build and return the argument parser for CLI invocation.

    All arguments have sensible defaults loaded from .env so the script
    can run with zero flags if .env is configured.
    """
    parser = argparse.ArgumentParser(
        description="Extract face crops from videos/images for anti-spoofing.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=DEFAULT_DATASET_ROOT,
        help="Root directory of the raw dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Base output directory for cropped frames.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_SAMPLE_FPS,
        help="Frames per second to sample from videos (1-5).",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        default=DEFAULT_CROP_SIZE,
        help="Square pixel size for each face crop.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default=DEFAULT_LOG_LEVEL,
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════
# 10. Main pipeline
# ═══════════════════════════════════════════════════════════════
def main() -> None:
    """
    Entry point: orchestrate the full face-extraction pipeline.

    Steps:
        1.  Parse CLI args / .env config.
        2.  Write architecture.md.
        3.  Set up logging.
        4.  Initialise MTCNN detector.
        5.  Discover video & image files under dataset_root.
        6.  For each file — detect, crop, save, record.
        7.  Write labels.csv.
    """
    # ── Parse arguments ──────────────────────────────────────
    args = parse_args()

    # ── Setup logging early ──────────────────────────────────
    logger = setup_logging(args.log_level)

    # ── Write/update architecture.md ─────────────────────────
    project_root = os.path.dirname(os.path.abspath(__file__))
    write_architecture_doc(project_root)

    # ── Validate dataset root ────────────────────────────────
    if not args.dataset_root or not os.path.isdir(args.dataset_root):
        logger.error(
            "DATASET_ROOT is not set or does not exist: '%s'. "
            "Set it in .env or pass --dataset_root.",
            args.dataset_root,
        )
        sys.exit(1)

    # ── Clamp FPS to 1-5 range ───────────────────────────────
    sample_fps = max(1, min(5, args.fps))
    if sample_fps != args.fps:
        logger.warning("FPS clamped to %d (requested %d)", sample_fps, args.fps)

    # ── Derive dataset name from the root folder ─────────────
    dataset_name = os.path.basename(os.path.normpath(args.dataset_root))

    # ── Build output directories ─────────────────────────────
    real_dir  = os.path.join(args.output_dir, dataset_name, "real")
    spoof_dir = os.path.join(args.output_dir, dataset_name, "spoof")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(spoof_dir, exist_ok=True)
    logger.info("Output dirs: %s , %s", real_dir, spoof_dir)

    # ── Initialise face detector ─────────────────────────────
    detector = create_detector(crop_size=args.crop_size)

    # ── Discover files ───────────────────────────────────────
    videos, images = discover_files(args.dataset_root)

    if not videos and not images:
        logger.error("No video or image files found under %s", args.dataset_root)
        sys.exit(1)

    # ── Counters for unique filenames per label ──────────────
    counters = {"real": 0, "spoof": 0, "unknown": 0}
    all_records: List[dict] = []
    total_saved = 0

    # ── Process videos ───────────────────────────────────────
    for vpath in videos:
        label = infer_label(vpath)
        out_dir = real_dir if label == "real" else spoof_dir

        try:
            saved, records = extract_faces_from_video(
                video_path=vpath,
                detector=detector,
                output_dir=out_dir,
                label=label,
                sample_fps=sample_fps,
                crop_size=args.crop_size,
                counter_start=counters[label],
            )
        except Exception as exc:
            logger.error("Failed on video %s: %s — skipping", vpath, exc)
            continue

        counters[label] += saved
        total_saved += saved
        all_records.extend(records)

    # ── Process images ───────────────────────────────────────
    for ipath in images:
        label = infer_label(ipath)
        out_dir = real_dir if label == "real" else spoof_dir

        try:
            saved, records = extract_faces_from_image(
                image_path=ipath,
                detector=detector,
                output_dir=out_dir,
                label=label,
                crop_size=args.crop_size,
                counter_start=counters[label],
            )
        except Exception as exc:
            logger.error("Failed on image %s: %s — skipping", ipath, exc)
            continue

        counters[label] += saved
        total_saved += saved
        all_records.extend(records)

    # ── Write CSV ────────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, dataset_name, "labels.csv")
    write_csv(all_records, csv_path)

    # ── Summary ──────────────────────────────────────────────
    logger.info("═" * 50)
    logger.info("Pipeline complete.")
    logger.info("  Total crops saved : %d", total_saved)
    logger.info("  Real              : %d", counters["real"])
    logger.info("  Spoof             : %d", counters["spoof"])
    logger.info("  Unknown-label     : %d", counters["unknown"])
    logger.info("  CSV               : %s", csv_path)
    logger.info("═" * 50)


# ═══════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
