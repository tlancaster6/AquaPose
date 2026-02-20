"""Integration tests for the YOLO -> SAM2 -> Label Studio export pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import (
    MagicMock,
    patch,
)

import cv2
import numpy as np
import pytest

from aquapose.segmentation import (
    SAMPseudoLabeler,
    export_to_label_studio,
    make_detector,
)
from aquapose.segmentation.detector import Detection
from aquapose.segmentation.pseudo_labeler import FrameAnnotation

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_yolo_mock_cls(
    boxes_xyxy: list[tuple[float, float, float, float]],
    confs: list[float],
) -> MagicMock:
    """Build a mock ultralytics YOLO class returning the given boxes on predict().

    Args:
        boxes_xyxy: List of (x1, y1, x2, y2) bounding boxes.
        confs: Confidence scores corresponding to each box.

    Returns:
        MagicMock acting as the YOLO class constructor.
    """
    mock_yolo_cls = MagicMock()
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model

    mock_boxes = []
    for (x1, y1, x2, y2), conf in zip(boxes_xyxy, confs, strict=True):
        box = MagicMock()
        box.xyxy = [MagicMock()]
        box.xyxy[0].tolist.return_value = [x1, y1, x2, y2]
        box.conf = [MagicMock()]
        box.conf[0].__float__ = MagicMock(return_value=conf)
        mock_boxes.append(box)

    mock_result = MagicMock()
    mock_result.boxes = mock_boxes
    mock_model.predict.return_value = [mock_result]

    return mock_yolo_cls


# ---------------------------------------------------------------------------
# Test 1: YOLO detections feed into SAMPseudoLabeler.predict()
# ---------------------------------------------------------------------------


class TestYOLODetectionsFeedIntoSAMPredict:
    """YOLO Detection objects are accepted and processed by SAMPseudoLabeler."""

    def test_yolo_detections_feed_into_sam_predict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mock YOLODetector and SAMPseudoLabeler.predict() to verify compatibility.

        Creates a Detection object matching YOLO output format and confirms that
        SAMPseudoLabeler.predict() accepts it and returns properly shaped masks.
        """
        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)

        # Create a Detection that mimics YOLODetector output
        full_frame_mask = np.zeros((480, 640), dtype=np.uint8)
        full_frame_mask[100:200, 100:200] = 255
        detection = Detection(
            bbox=(100, 100, 100, 100),
            mask=full_frame_mask,
            area=40000,
            confidence=0.85,
        )

        # Build expected mask output (simulating SAM2 refinement)
        expected_mask = np.zeros((480, 640), dtype=np.uint8)
        expected_mask[110:190, 110:190] = 255

        # Monkeypatch SAMPseudoLabeler.predict to return fake masks without GPU
        def fake_predict(
            self_inner: SAMPseudoLabeler,
            image: np.ndarray,
            detections: list[Detection],
            use_mask_prompt: bool = True,
        ) -> list[np.ndarray]:
            return [expected_mask.copy() for _ in detections]

        monkeypatch.setattr(SAMPseudoLabeler, "predict", fake_predict)

        labeler = SAMPseudoLabeler()
        result_masks = labeler.predict(fake_image, [detection])

        assert isinstance(result_masks, list)
        assert len(result_masks) == 1
        assert result_masks[0].shape == (480, 640)
        unique_vals = set(np.unique(result_masks[0]))
        assert unique_vals <= {0, 255}


# ---------------------------------------------------------------------------
# Test 2: make_detector("yolo") produces Detection objects with correct format
# ---------------------------------------------------------------------------


class TestMakeDetectorYOLOProducesCompatibleDetections:
    """make_detector('yolo', ...) returns a detector producing valid Detection objects."""

    def test_make_detector_yolo_produces_compatible_detections(self) -> None:
        """YOLODetector from make_detector produces Detection objects with correct fields.

        Monkeypatches sys.modules to avoid requiring ultralytics. Verifies that
        each Detection has the expected field types and constraints.
        """
        mock_yolo_cls = _make_yolo_mock_cls(
            boxes_xyxy=[(100.0, 200.0, 300.0, 400.0)],
            confs=[0.87],
        )

        with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_yolo_cls)}):
            detector = make_detector("yolo", model_path="dummy.pt")

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)

        assert len(detections) == 1
        det = detections[0]

        # bbox is a tuple of 4 ints
        assert isinstance(det.bbox, tuple)
        assert len(det.bbox) == 4
        assert all(isinstance(v, int) for v in det.bbox)

        # mask is full-frame shaped ndarray
        assert isinstance(det.mask, np.ndarray)
        assert det.mask.shape == (480, 640)

        # confidence is a float in [0, 1]
        assert isinstance(det.confidence, float)
        assert 0.0 <= det.confidence <= 1.0

        # area is a positive int
        assert isinstance(det.area, int)
        assert det.area > 0


# ---------------------------------------------------------------------------
# Test 3: Full pipeline from detector to export
# ---------------------------------------------------------------------------


class TestFullPipelineDetectorToExport:
    """End-to-end chain: make_detector -> detect -> predict -> FrameAnnotation -> export."""

    def test_full_pipeline_detector_to_export(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Mock detector and SAM2 and run the full pipeline to Label Studio JSON.

        Verifies that the tasks JSON is written, contains the expected number of
        tasks, and each task has the required data fields and predictions structure.
        """
        # --- Mock YOLO detector ---
        mock_yolo_cls = _make_yolo_mock_cls(
            boxes_xyxy=[(50.0, 60.0, 200.0, 250.0)],
            confs=[0.9],
        )

        # --- Mock SAMPseudoLabeler.predict (no GPU needed) ---
        fake_mask = np.zeros((480, 640), dtype=np.uint8)
        fake_mask[60:250, 50:200] = 255

        def fake_predict(
            self_inner: SAMPseudoLabeler,
            image: np.ndarray,
            detections: list[Detection],
            use_mask_prompt: bool = True,
        ) -> list[np.ndarray]:
            return [fake_mask.copy() for _ in detections]

        monkeypatch.setattr(SAMPseudoLabeler, "predict", fake_predict)

        # --- Build fake image and save it to tmp_path ---
        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        image_path = tmp_path / "images" / "cam1_frame_000100.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image_path), fake_image)

        # --- Run full chain ---
        with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_yolo_cls)}):
            detector = make_detector("yolo", model_path="dummy.pt")

        detections = detector.detect(fake_image)

        labeler = SAMPseudoLabeler()
        masks = labeler.predict(fake_image, detections)

        annotation = FrameAnnotation(
            frame_id="cam1_frame_000100",
            image_path=image_path,
            masks=masks,
            camera_id="cam1",
        )

        output_dir = tmp_path / "output"
        tasks_path = export_to_label_studio([annotation], output_dir)

        # --- Assertions on output ---
        assert tasks_path.exists(), "Tasks JSON was not written"

        with open(tasks_path) as f:
            tasks = json.load(f)

        assert len(tasks) == 1, f"Expected 1 task, got {len(tasks)}"

        task = tasks[0]
        assert "data" in task
        data = task["data"]
        assert "image" in data
        assert "frame_id" in data
        assert "camera_id" in data
        assert data["frame_id"] == "cam1_frame_000100"
        assert data["camera_id"] == "cam1"

        # Task has masks -> should have predictions with results
        assert "predictions" in task
        assert len(task["predictions"]) == 1
        results = task["predictions"][0]["result"]
        assert len(results) >= 1, "Expected at least one brush result"
