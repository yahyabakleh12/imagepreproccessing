import argparse
import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("plate_cropper_for_preprocessing")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
GENERATED_DIR_NAMES = {
    "preprocessing_results",
    "lpd",
    "filtered_lpd",
    "filters",
    "__pycache__",
    "diagrams",
}


def resolve_model_path() -> Path:
    candidates = []

    env_path = os.getenv("PLATE_MODEL_PATH")
    if env_path:
        candidates.append(Path(env_path))

    candidates.extend(
        [
            Path("models/license_plate_detector_int8_openvino_model"),
            Path("license_plate_detector_int8_openvino_model"),
        ]
    )

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find license plate detector model. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def load_plate_model():
    from ultralytics import YOLO

    model_path = resolve_model_path()
    logger.info("Loading plate detector model from: %s", model_path)
    return YOLO(str(model_path), task="detect")


def is_inside_path(path: Path, folder: Path) -> bool:
    try:
        path.resolve().relative_to(folder.resolve())
        return True
    except ValueError:
        return False


def is_generated_folder_name(name: str) -> bool:
    lowered = name.lower()
    return lowered in GENERATED_DIR_NAMES or lowered.startswith("folder_test_results")


def collect_image_paths(input_path: str, output_root: Optional[Path] = None) -> List[Path]:
    path = Path(input_path)

    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"File is not a supported image: {path}")
        return [path]

    if not path.is_dir():
        raise FileNotFoundError(f"Input path not found: {path}")

    image_paths = []
    for child in path.rglob("*"):
        if not child.is_file() or child.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if output_root is not None and is_inside_path(child, output_root):
            continue

        try:
            relative_parts = child.relative_to(path).parts[:-1]
        except ValueError:
            relative_parts = child.parts[:-1]

        if any(is_generated_folder_name(part) for part in relative_parts):
            continue

        image_paths.append(child)

    image_paths = sorted(image_paths)
    if not image_paths:
        raise FileNotFoundError(f"No supported image files found in folder: {path}")
    return image_paths


def resolve_one_input_path(args) -> str:
    provided_paths = [value for value in (args.input_path, args.image, args.folder) if value]
    if len(provided_paths) != 1:
        raise ValueError("Provide exactly one path: input_path, --image, or --folder.")
    return provided_paths[0]


def default_output_root(input_path: str) -> Path:
    path = Path(input_path)
    base = path if path.is_dir() else path.parent
    return base / "preprocessing_results"


def result_folder_for_image(output_root: Path, image_path: Path) -> Path:
    return output_root / image_path.name


def original_output_name(image_path: Path) -> str:
    suffix = image_path.suffix.lower() or ".jpg"
    return f"original_image{suffix}"


def lpd_output_name(index: int) -> str:
    return "lpd.jpg" if index == 0 else f"lpd_{index + 1}.jpg"


def copy_original_image(image_path: Path, result_dir: Path) -> str:
    result_dir.mkdir(parents=True, exist_ok=True)
    output_path = result_dir / original_output_name(image_path)
    shutil.copy2(str(image_path), str(output_path))
    return str(output_path.resolve())


def safe_stem(path: Path) -> str:
    stem = path.stem.strip() or "image"
    return re.sub(r"[^A-Za-z0-9._-]", "_", stem)


def clamp_bbox(xyxy, image_width: int, image_height: int) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy]
    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(0, min(x2, image_width))
    y2 = max(0, min(y2, image_height))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def crop_dimensions(x1: int, y1: int, x2: int, y2: int) -> Dict[str, int]:
    return {
        "x": x1,
        "y": y1,
        "width": x2 - x1,
        "height": y2 - y1,
    }


def bbox_record_for_detection(
    image_path: Path,
    xyxy,
    index: int,
    crop_path: Optional[str],
    clamped_bbox: Optional[Tuple[int, int, int, int]],
    confidence=None,
    class_id=None,
) -> Dict[str, Any]:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    record: Dict[str, Any] = {
        "image": str(image_path),
        "index": index,
        "raw_lpd": crop_path,
        "plate_crop": crop_path,
        "lpd_after_preprocessing": None,
        "pasted_to_original": False,
        "bbox_xyxy": {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        },
        "crop_dimensions": None,
    }

    if clamped_bbox is not None:
        record["crop_dimensions"] = crop_dimensions(*clamped_bbox)
    if confidence is not None:
        record["confidence"] = float(confidence)
    if class_id is not None:
        record["class_id"] = int(float(class_id))
    return record


def save_plate_crop_to_dir(
    frame,
    source_image_path: Path,
    output_dir: Path,
    xyxy,
    output_name: str,
) -> Tuple[Optional[str], Optional[Tuple[int, int, int, int]]]:
    h, w = frame.shape[:2]
    clamped_bbox = clamp_bbox(xyxy, w, h)
    if clamped_bbox is None:
        logger.warning("Invalid plate bbox for image=%s bbox=%s", source_image_path, xyxy)
        return None, None

    x1, y1, x2, y2 = clamped_bbox
    plate_crop = frame[y1:y2, x1:x2]
    if plate_crop.size == 0:
        logger.warning("Empty plate crop for image=%s bbox=%s", source_image_path, xyxy)
        return None, None

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name
    if not cv2.imwrite(str(output_path), plate_crop):
        logger.warning("Failed to save plate crop: %s", output_path)
        return None, None

    logger.info("Saved raw LPD crop: %s", output_path)
    return str(output_path.resolve()), clamped_bbox


def detect_and_crop_plates(
    model,
    image_path: Path,
    result_dir: Path,
    conf: float,
    iou: float,
) -> Tuple[bool, List[Dict[str, Any]], int, float]:
    frame = cv2.imread(str(image_path))
    if frame is None or frame.size == 0:
        raise ValueError(f"Image is empty or unreadable: {image_path}")

    start = time.perf_counter()
    results = model(frame, conf=conf, iou=iou, verbose=False)[0]
    duration = time.perf_counter() - start

    xyxy_list = results.boxes.xyxy if results.boxes is not None else []
    conf_list = results.boxes.conf if results.boxes is not None else []
    cls_list = results.boxes.cls if results.boxes is not None else []
    detected_count = len(xyxy_list)
    plate_records = []

    for index, xyxy in enumerate(xyxy_list):
        crop_path, clamped_bbox = save_plate_crop_to_dir(
            frame,
            image_path,
            result_dir,
            xyxy,
            lpd_output_name(index),
        )
        confidence = conf_list[index] if index < len(conf_list) else None
        class_id = cls_list[index] if index < len(cls_list) else None
        plate_records.append(
            bbox_record_for_detection(
                image_path,
                xyxy,
                index,
                crop_path,
                clamped_bbox,
                confidence=confidence,
                class_id=class_id,
            )
        )

    return detected_count > 0, plate_records, detected_count, duration


def save_json(data: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def process_one_image(
    model,
    image_path: Path,
    output_root: Path,
    conf: float,
    iou: float,
) -> Dict[str, Any]:
    image_path = image_path.resolve()
    result_dir = result_folder_for_image(output_root, image_path)
    result_dir.mkdir(parents=True, exist_ok=True)

    record: Dict[str, Any] = {
        "pipeline_stage": "plate_cropper_for_preprocessing",
        "image": str(image_path),
        "source_image": str(image_path),
        "image_stem": safe_stem(image_path),
        "result_folder": str(result_dir.resolve()),
        "original_image": None,
        "has_plate": False,
        "detected_plate_count": 0,
        "lpd_crops": [],
        "image_after_preprocessing": None,
        "pasted_filtered_lpd_count": 0,
        "error": None,
    }

    try:
        record["original_image"] = copy_original_image(image_path, result_dir)
        has_plate, plate_records, detected_count, duration = detect_and_crop_plates(
            model,
            image_path,
            result_dir,
            conf,
            iou,
        )
        record["has_plate"] = has_plate
        record["detected_plate_count"] = detected_count
        record["plate_detector_time_sec"] = round(duration, 3)
        record["lpd_crops"] = plate_records
    except Exception as exc:
        record["error"] = str(exc)
        logger.error("Failed image=%s error=%s", image_path, exc)

    save_json(record, result_dir / "result.json")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 1: detect license plates, crop them, and save crop dimensions "
            "for the preprocessing paste-back stage."
        )
    )
    parser.add_argument("input_path", nargs="?", help="Path to one image or a folder of images.")
    parser.add_argument("--image", help="Path to one image file.")
    parser.add_argument("--folder", help="Path to a folder containing images.")
    parser.add_argument(
        "--output-root",
        help="Root folder for per-image results. Default: <input>/preprocessing_results",
    )
    parser.add_argument("--conf", type=float, default=0.15, help="Plate detector confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.3, help="Plate detector IoU threshold.")
    args = parser.parse_args()

    try:
        input_path = resolve_one_input_path(args)
        output_root = Path(args.output_root).resolve() if args.output_root else default_output_root(input_path).resolve()
        image_paths = collect_image_paths(input_path, output_root)
        model = load_plate_model()
    except Exception as exc:
        logger.error("%s", exc)
        return 1

    records = []
    for image_path in image_paths:
        record = process_one_image(model, image_path, output_root, args.conf, args.iou)
        records.append(record)

        print(f"image: {image_path}")
        print(f"folder: {record['result_folder']}")
        print(f"has_plate: {record['has_plate']}")
        print(f"cropped_plates: {len(record['lpd_crops'])}")
        if record["error"]:
            print(f"error: {record['error']}")
        print()

    print("summary:")
    print(f"  output_root: {output_root}")
    print(f"  images_tested: {len(records)}")
    print(f"  images_with_plate: {sum(1 for item in records if item['has_plate'])}")
    failed_count = sum(1 for item in records if item["error"])
    print(f"  failed_images: {failed_count}")
    return 1 if failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
