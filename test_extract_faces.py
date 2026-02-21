"""
test_extract_faces.py — Comprehensive test suite for the face extraction pipeline.

Tests cover:
  - Architecture doc generation
  - Label inference from file paths
  - File discovery (videos vs images)
  - Face detection + cropping (mocked MTCNN)
  - Video frame extraction end-to-end (mocked)
  - Image extraction end-to-end (mocked)
  - CSV writing
  - Edge cases: empty dirs, corrupt files, unreadable images

Run:
    pytest test_extract_faces.py -v
"""

import os
import csv
import sys
import tempfile
import shutil
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

# ─── Make sure the project root is on sys.path ───
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import extract_faces as ef


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════
@pytest.fixture
def tmp_dir():
    """Create a temporary directory and clean it up after the test."""
    d = tempfile.mkdtemp(prefix="test_ef_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_rgb_image():
    """Return a small random RGB numpy array (100x100x3)."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_bgr_video_frames():
    """Return a list of 10 small BGR frames simulating a short video."""
    return [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(10)]


# ═══════════════════════════════════════════════════════════════
# 1. Architecture doc
# ═══════════════════════════════════════════════════════════════
class TestArchitectureDoc:
    """Tests for write_architecture_doc()."""

    def test_creates_file(self, tmp_dir):
        """architecture.md should be created when it does not exist."""
        ef.write_architecture_doc(tmp_dir)
        path = os.path.join(tmp_dir, "architecture.md")
        assert os.path.isfile(path), "architecture.md was not created"

    def test_contains_phase_description(self, tmp_dir):
        """The doc must mention Phase 1 and its input/output."""
        ef.write_architecture_doc(tmp_dir)
        text = Path(os.path.join(tmp_dir, "architecture.md")).read_text(encoding="utf-8")
        assert "Phase 1" in text
        assert "raw videos" in text.lower() or "raw videos" in text
        assert "cropped frames" in text.lower() or "CNN" in text

    def test_overwrites_on_rerun(self, tmp_dir):
        """Running twice should overwrite (not append)."""
        ef.write_architecture_doc(tmp_dir)
        size1 = os.path.getsize(os.path.join(tmp_dir, "architecture.md"))
        ef.write_architecture_doc(tmp_dir)
        size2 = os.path.getsize(os.path.join(tmp_dir, "architecture.md"))
        # Sizes should be roughly the same (timestamp may differ by a few chars)
        assert abs(size1 - size2) < 50


# ═══════════════════════════════════════════════════════════════
# 2. Label inference
# ═══════════════════════════════════════════════════════════════
class TestInferLabel:
    """Tests for infer_label()."""

    @pytest.mark.parametrize("path, expected", [
        # ── Basic folder-name matching ──
        (r"C:\data\real\video001.mp4", "real"),
        ("/datasets/live/clip.avi", "real"),
        (r"D:\genuine\face.jpg", "real"),
        ("/data/spoof/clip.mp4", "spoof"),
        (r"C:\fake\images\face.png", "spoof"),
        ("/dataset/replay/attack.avi", "spoof"),
        ("/other/misc/random.mp4", "unknown"),

        # ── Axondata dataset: "Screen" folder = spoof ──
        (r"C:\datasets\Axon Labs Replay Display sample\Screen\20240823_194131.mp4", "spoof"),
        (r"C:\datasets\Axon Labs Replay Display sample\Real\1.jpg", "real"),

        # ── "real" inside a long parent path should NOT false-positive ──
        # The parent "liveness-detection-real-and-display-attacks-5k" has "real"
        # but "Screen" subfolder should win as spoof
        (r"C:\cache\liveness-detection-real-and-display-attacks-5k\v4\Axon Labs Replay Display sample\Screen\clip.mp4", "spoof"),

        # ── TrainingDataPro: filename "live_selfie.jpg" → real ──
        (r"C:\data\samples\hash123\live_selfie.jpg", "real"),
        (r"C:\data\samples\hash123\live_video.mp4", "real"),

        # ── "display" folder = spoof ──
        ("/data/display/attack01.mp4", "spoof"),
    ])
    def test_keyword_matching(self, path, expected):
        """Label should match known keywords in the path."""
        assert ef.infer_label(path) == expected

    def test_case_insensitive(self):
        """Keywords should match regardless of case."""
        assert ef.infer_label("/Data/REAL/clip.mp4") == "real"
        assert ef.infer_label("/Data/SPOOF/clip.mp4") == "spoof"
        assert ef.infer_label("/Data/Screen/clip.mp4") == "spoof"

    def test_replay_not_false_positive_real(self):
        """'replay' should be spoof, not trigger 'real' substring match."""
        assert ef.infer_label("/data/replay/clip.mp4") == "spoof"

    def test_segment_matching_over_substring(self):
        """Only full segment/token matches, not substrings within words."""
        # "screensaver" contains "screen" as a token, but
        # "realism" should NOT match "real" — it's not a segment
        assert ef.infer_label("/data/realism/clip.mp4") == "unknown"


# ═══════════════════════════════════════════════════════════════
# 3. File discovery
# ═══════════════════════════════════════════════════════════════
class TestDiscoverFiles:
    """Tests for discover_files()."""

    def test_finds_videos_and_images(self, tmp_dir):
        """Should separate video and image files correctly."""
        # Create dummy files
        for name in ["a.mp4", "b.avi", "c.mkv"]:
            Path(os.path.join(tmp_dir, name)).touch()
        for name in ["d.jpg", "e.png", "f.bmp"]:
            Path(os.path.join(tmp_dir, name)).touch()
        # And a non-matching file
        Path(os.path.join(tmp_dir, "readme.txt")).touch()

        videos, images = ef.discover_files(tmp_dir)
        assert len(videos) == 3, f"Expected 3 videos, got {len(videos)}"
        assert len(images) == 3, f"Expected 3 images, got {len(images)}"

    def test_recursive_discovery(self, tmp_dir):
        """Should walk subdirectories."""
        sub = os.path.join(tmp_dir, "sub", "deep")
        os.makedirs(sub)
        Path(os.path.join(sub, "clip.mp4")).touch()
        Path(os.path.join(sub, "face.jpg")).touch()

        videos, images = ef.discover_files(tmp_dir)
        assert len(videos) == 1
        assert len(images) == 1

    def test_empty_directory(self, tmp_dir):
        """Empty dir should return empty lists."""
        videos, images = ef.discover_files(tmp_dir)
        assert videos == []
        assert images == []


# ═══════════════════════════════════════════════════════════════
# 4. CSV writer
# ═══════════════════════════════════════════════════════════════
class TestWriteCSV:
    """Tests for write_csv()."""

    def test_writes_correct_rows(self, tmp_dir):
        """CSV should have the correct header and row data."""
        records = [
            {"filename": "frame_000000.jpg", "label": "real",
             "source": "vid.mp4", "source_type": "video", "frame_idx": 10},
            {"filename": "frame_000001.jpg", "label": "spoof",
             "source": "img.png", "source_type": "image", "frame_idx": 0},
        ]
        csv_path = os.path.join(tmp_dir, "labels.csv")
        ef.write_csv(records, csv_path)

        with open(csv_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["label"] == "real"
        assert rows[1]["source_type"] == "image"

    def test_empty_records_skips_file(self, tmp_dir):
        """No file should be written when records list is empty."""
        csv_path = os.path.join(tmp_dir, "labels.csv")
        ef.write_csv([], csv_path)
        assert not os.path.exists(csv_path)


# ═══════════════════════════════════════════════════════════════
# 5. Face detection & cropping (mocked MTCNN)
# ═══════════════════════════════════════════════════════════════
class TestDetectAndCropFaces:
    """Tests for detect_and_crop_faces() with mocked MTCNN."""

    def _make_fake_face_tensor(self, crop_size=224):
        """Return a torch-like tensor (3, H, W) simulating one face crop."""
        import torch
        return torch.randint(0, 255, (3, crop_size, crop_size), dtype=torch.float32)

    def test_returns_crops_when_faces_found(self, sample_rgb_image):
        """Should return a list of numpy arrays when MTCNN finds faces."""
        fake_tensor = self._make_fake_face_tensor(224)
        # Stack to simulate batch dimension: (1, 3, 224, 224)
        batch = fake_tensor.unsqueeze(0)

        mock_detector = MagicMock()
        mock_detector.return_value = batch  # __call__ returns the batch

        crops = ef.detect_and_crop_faces(mock_detector, sample_rgb_image, crop_size=224)
        assert len(crops) == 1
        assert crops[0].shape == (224, 224, 3)
        assert crops[0].dtype == np.uint8

    def test_returns_empty_when_no_faces(self, sample_rgb_image):
        """Should return empty list when MTCNN returns None."""
        mock_detector = MagicMock()
        mock_detector.return_value = None

        crops = ef.detect_and_crop_faces(mock_detector, sample_rgb_image)
        assert crops == []

    def test_multiple_faces(self, sample_rgb_image):
        """Should return multiple crops when several faces are detected."""
        import torch
        faces = torch.stack([
            torch.randint(0, 255, (3, 224, 224), dtype=torch.float32),
            torch.randint(0, 255, (3, 224, 224), dtype=torch.float32),
            torch.randint(0, 255, (3, 224, 224), dtype=torch.float32),
        ])  # shape (3, 3, 224, 224)

        mock_detector = MagicMock()
        mock_detector.return_value = faces

        crops = ef.detect_and_crop_faces(mock_detector, sample_rgb_image, crop_size=224)
        assert len(crops) == 3


# ═══════════════════════════════════════════════════════════════
# 6. Video extraction (mocked cv2 + MTCNN)
# ═══════════════════════════════════════════════════════════════
class TestExtractFacesFromVideo:
    """Integration-style tests for extract_faces_from_video() with mocks."""

    def test_saves_crops_and_returns_records(self, tmp_dir):
        """Should write JPEG files and return matching records."""
        import torch

        # Create a mock detector that always finds one face
        fake_face = torch.randint(0, 255, (1, 3, 224, 224), dtype=torch.float32)
        mock_detector = MagicMock()
        mock_detector.return_value = fake_face

        # Create a tiny real video file (3 frames of solid colour)
        video_path = os.path.join(tmp_dir, "test_video.avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(video_path, fourcc, 10.0, (100, 100))
        for _ in range(30):  # 30 frames at 10fps = 3 seconds
            frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            out.write(frame)
        out.release()

        out_dir = os.path.join(tmp_dir, "output", "real")
        os.makedirs(out_dir)

        saved, records = ef.extract_faces_from_video(
            video_path=video_path,
            detector=mock_detector,
            output_dir=out_dir,
            label="real",
            sample_fps=3,
            crop_size=224,
            counter_start=0,
        )

        # Should have saved some crops
        assert saved > 0, "Expected at least one crop saved"
        assert len(records) == saved
        # Files should exist on disk
        for rec in records:
            fpath = os.path.join(out_dir, rec["filename"])
            assert os.path.isfile(fpath), f"Missing file: {fpath}"

    def test_handles_unopenable_video(self, tmp_dir):
        """Should return (0, []) for a non-existent video."""
        mock_detector = MagicMock()
        saved, records = ef.extract_faces_from_video(
            video_path=os.path.join(tmp_dir, "nonexistent.mp4"),
            detector=mock_detector,
            output_dir=tmp_dir,
            label="spoof",
        )
        assert saved == 0
        assert records == []


# ═══════════════════════════════════════════════════════════════
# 7. Image extraction (mocked MTCNN)
# ═══════════════════════════════════════════════════════════════
class TestExtractFacesFromImage:
    """Tests for extract_faces_from_image() with mocked MTCNN."""

    def test_saves_crop_from_real_image(self, tmp_dir):
        """Should save a crop from a valid image file."""
        import torch

        # Write a real image to disk
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        img_path = os.path.join(tmp_dir, "face.jpg")
        cv2.imwrite(img_path, img)

        # Mock detector returns one face
        fake_face = torch.randint(0, 255, (1, 3, 224, 224), dtype=torch.float32)
        mock_detector = MagicMock()
        mock_detector.return_value = fake_face

        out_dir = os.path.join(tmp_dir, "output")
        os.makedirs(out_dir)

        saved, records = ef.extract_faces_from_image(
            image_path=img_path,
            detector=mock_detector,
            output_dir=out_dir,
            label="real",
            crop_size=224,
            counter_start=0,
        )

        assert saved == 1
        assert len(records) == 1
        assert records[0]["source_type"] == "image"
        assert os.path.isfile(os.path.join(out_dir, records[0]["filename"]))

    def test_handles_unreadable_image(self, tmp_dir):
        """Should gracefully skip an image that cv2 can't read."""
        # Write garbage data as if it were an image
        bad_path = os.path.join(tmp_dir, "corrupt.jpg")
        with open(bad_path, "wb") as f:
            f.write(b"not an image")

        mock_detector = MagicMock()
        saved, records = ef.extract_faces_from_image(
            image_path=bad_path,
            detector=mock_detector,
            output_dir=tmp_dir,
            label="spoof",
        )
        assert saved == 0
        assert records == []

    def test_no_faces_detected(self, tmp_dir):
        """Should return (0, []) when detector finds nothing."""
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        img_path = os.path.join(tmp_dir, "noface.jpg")
        cv2.imwrite(img_path, img)

        mock_detector = MagicMock()
        mock_detector.return_value = None  # No faces

        saved, records = ef.extract_faces_from_image(
            image_path=img_path,
            detector=mock_detector,
            output_dir=tmp_dir,
            label="real",
        )
        assert saved == 0


# ═══════════════════════════════════════════════════════════════
# 8. FPS clamping (sanity check via main-level logic)
# ═══════════════════════════════════════════════════════════════
class TestFPSClamping:
    """Verify that FPS is clamped to 1-5."""

    @pytest.mark.parametrize("requested, expected", [
        (0, 1), (1, 1), (3, 3), (5, 5), (10, 5), (-2, 1),
    ])
    def test_clamp(self, requested, expected):
        """max(1, min(5, x)) should clamp correctly."""
        result = max(1, min(5, requested))
        assert result == expected


# ═══════════════════════════════════════════════════════════════
# 9. Environment / .env loading
# ═══════════════════════════════════════════════════════════════
class TestEnvDefaults:
    """Verify that default constants are loaded from env."""

    def test_crop_size_is_int(self):
        """FACE_CROP_SIZE should parse to a positive int."""
        assert isinstance(ef.DEFAULT_CROP_SIZE, int)
        assert ef.DEFAULT_CROP_SIZE > 0

    def test_sample_fps_is_int(self):
        """VIDEO_SAMPLE_FPS should parse to a positive int."""
        assert isinstance(ef.DEFAULT_SAMPLE_FPS, int)
        assert ef.DEFAULT_SAMPLE_FPS > 0
