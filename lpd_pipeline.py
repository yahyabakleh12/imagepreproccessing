"""Simple license-plate detection, enhancement, and paste-back pipeline."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

import cv2
from ultralytics import YOLO

from v1 import EnhancementPipelineV1


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent
    / "license_plate_detector_int8_openvino_model"
)


def find_images(input_path: str | Path, output_dir: str | Path | None = None) -> list[Path]:
    """Return one image or all images below a directory."""
    source = Path(input_path).expanduser().resolve()
    excluded = Path(output_dir).resolve() if output_dir else None

    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image type: {source.suffix}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {source}")

    images = []
    for path in source.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        resolved = path.resolve()
        if excluded and (resolved == excluded or excluded in resolved.parents):
            continue
        images.append(resolved)
    if not images:
        raise FileNotFoundError(f"No supported images found in: {source}")
    return sorted(images)


def _bbox(values: Iterable[float], width: int, height: int) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = (int(round(float(value))) for value in values)
    x1, y1 = max(0, min(x1, width - 1)), max(0, min(y1, height - 1))
    x2, y2 = max(0, min(x2, width)), max(0, min(y2, height))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


class PlatePipeline:
    """Reusable pipeline. Create once so the detector model is loaded once."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        confidence: float = 0.15,
        iou: float = 0.30,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Detector model not found: {self.model_path}")
        self.confidence = confidence
        self.iou = iou
        self.model = YOLO(str(self.model_path), task="detect")
        self.plate_enhancer = EnhancementPipelineV1()

    def process_image(self, image_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
        """Process one car image and return paths plus plate dimensions."""
        source = Path(image_path).expanduser().resolve()
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError(f"Image is empty or unreadable: {source}")

        result_dir = Path(output_dir).expanduser().resolve() / source.stem
        result_dir.mkdir(parents=True, exist_ok=True)
        # Prevent stale crops from an earlier run (including the former
        # ``*_filtered.png`` naming) from being mistaken for current results.
        for pattern in (
            "plate_*_raw.png",
            "plate_*_enhanced.png",
            "plate_*_stage_*.png",
            "plate_*_filtered.png",
            "original.*",
        ):
            for old_crop in result_dir.glob(pattern):
                old_crop.unlink()

        original_path = result_dir / f"original{source.suffix.lower() or '.jpg'}"
        shutil.copy2(source, original_path)
        height, width = image.shape[:2]

        started = time.perf_counter()
        prediction = self.model(
            image, conf=self.confidence, iou=self.iou, verbose=False
        )[0]
        detection_seconds = round(time.perf_counter() - started, 3)
        boxes = prediction.boxes

        final_image = image.copy()
        plates: list[dict[str, Any]] = []
        if boxes is not None:
            for index, coordinates in enumerate(boxes.xyxy):
                bounds = _bbox(coordinates, width, height)
                if bounds is None:
                    continue
                x1, y1, x2, y2 = bounds
                raw_crop = image[y1:y2, x1:x2]

                raw_path = result_dir / f"plate_{index + 1}_raw.png"
                enhanced_path = result_dir / f"plate_{index + 1}_enhanced.png"
                if not cv2.imwrite(str(raw_path), raw_crop):
                    raise RuntimeError(f"Could not save crop: {raw_path}")

                # Save the detector crop before the slower v1 enhancement stages.
                # This gives callers an immediate artifact and preserves useful
                # detector output if enhancement is interrupted.
                print(
                    f"  plate {index + 1}: saved raw crop; applying v1 enhancement...",
                    flush=True,
                )
                stage_outputs = self.plate_enhancer.apply_with_stages(raw_crop)
                stage_records: list[dict[str, Any]] = []
                for stage_index, (stage_name, stage_image) in enumerate(
                    stage_outputs, start=1
                ):
                    stage_slug = stage_name.lower().replace(" ", "_")
                    stage_path = result_dir / (
                        f"plate_{index + 1}_stage_{stage_index}_{stage_slug}.png"
                    )
                    stage_for_disk = (
                        cv2.cvtColor(stage_image, cv2.COLOR_RGB2BGR)
                        if stage_image.ndim == 3
                        else stage_image
                    )
                    if not cv2.imwrite(str(stage_path), stage_for_disk):
                        raise RuntimeError(f"Could not save stage output: {stage_path}")
                    stage_records.append(
                        {
                            "index": stage_index,
                            "name": stage_slug,
                            "label": stage_name,
                            "image": str(stage_path),
                            "dimensions": {
                                "width": int(stage_image.shape[1]),
                                "height": int(stage_image.shape[0]),
                            },
                        }
                    )

                enhanced_crop = stage_outputs[-1][1]
                enhanced_for_cv = (
                    cv2.cvtColor(enhanced_crop, cv2.COLOR_RGB2BGR)
                    if enhanced_crop.ndim == 3
                    else enhanced_crop
                )
                if not cv2.imwrite(str(enhanced_path), enhanced_for_cv):
                    raise RuntimeError(f"Could not save enhanced crop: {enhanced_path}")

                pasted = cv2.resize(
                    enhanced_for_cv,
                    (x2 - x1, y2 - y1),
                    interpolation=cv2.INTER_AREA,
                )
                if pasted.ndim == 2:
                    pasted = cv2.cvtColor(pasted, cv2.COLOR_GRAY2BGR)
                final_image[y1:y2, x1:x2] = pasted

                confidence = float(boxes.conf[index]) if boxes.conf is not None else None
                class_id = int(boxes.cls[index]) if boxes.cls is not None else None
                plates.append(
                    {
                        "index": index,
                        "confidence": confidence,
                        "class_id": class_id,
                        "bbox_xyxy": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                        "dimensions": {
                            "x": x1,
                            "y": y1,
                            "width": x2 - x1,
                            "height": y2 - y1,
                        },
                        "raw_crop": str(raw_path),
                        "enhanced_crop": str(enhanced_path),
                        "stages": stage_records,
                    }
                )

        final_path = result_dir / "final.jpg"
        if not cv2.imwrite(str(final_path), final_image):
            raise RuntimeError(f"Could not save final image: {final_path}")

        record: dict[str, Any] = {
            "source_image": str(source),
            "original_image": str(original_path),
            "final_image": str(final_path),
            "image_dimensions": {"width": width, "height": height},
            "detected_plate_count": len(plates),
            "detector_time_seconds": detection_seconds,
            "plates": plates,
        }
        metadata_path = result_dir / "result.json"
        metadata_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        record["metadata_file"] = str(metadata_path)
        return record

    def process(self, input_path: str | Path, output_dir: str | Path) -> list[dict[str, Any]]:
        """Process one image or a directory of images."""
        return [
            self.process_image(image_path, output_dir)
            for image_path in find_images(input_path, output_dir)
        ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect plates, run v1 enhancement, and paste them back."
    )
    parser.add_argument("input", help="Image file or folder of images")
    parser.add_argument("-o", "--output", default="output", help="Output directory")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="OpenVINO model directory")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.30, help="IoU threshold")
    args = parser.parse_args()

    try:
        output_dir = Path(args.output).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        image_paths = find_images(args.input, output_dir)
        print(f"Output directory: {output_dir}", flush=True)
        print(
            f"Found {len(image_paths)} image(s). V1 enhancement can take time on CPU.",
            flush=True,
        )
        pipeline = PlatePipeline(args.model, args.conf, args.iou)
        records = []
        for index, image_path in enumerate(image_paths, start=1):
            print(
                f"[{index}/{len(image_paths)}] Processing {image_path.name}...",
                flush=True,
            )
            records.append(pipeline.process_image(image_path, output_dir))
    except Exception as exc:
        parser.exit(1, f"Error: {exc}\n")

    print(f"Processed {len(records)} image(s). Output: {output_dir}")
    print(f"Detected {sum(item['detected_plate_count'] for item in records)} plate(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
