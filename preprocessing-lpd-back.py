import argparse
import contextlib
import io
import importlib.util
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("preprocessing_lpd_back")


def default_results_root() -> Path:
    for path in (Path("images/preprocessing_results"), Path("preprocessing_results")):
        if path.is_dir():
            return path
    return Path("images/preprocessing_results")


def collect_result_folders(input_path: Path) -> List[Path]:
    if input_path.is_file():
        if input_path.name != "result.json":
            raise ValueError(f"Expected a result.json file, got: {input_path}")
        return [input_path.parent]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Results path not found: {input_path}")

    if (input_path / "result.json").is_file():
        return [input_path]

    folders = sorted({path.parent for path in input_path.rglob("result.json")})
    if not folders:
        raise FileNotFoundError(f"No result.json files found under: {input_path}")
    return folders


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in: {path}")
    return data


def save_json(data: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_final_image_copy(
    data: Dict[str, Any],
    result_dir: Path,
    final_images_dir: Path,
) -> Optional[str]:
    final_path = resolve_saved_path(data.get("image_after_preprocessing"), result_dir)
    if final_path is None or not final_path.is_file():
        return None

    image_stem = str(data.get("image_stem") or result_dir.stem).strip() or "image"
    output_path = final_images_dir / f"{image_stem}.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(final_path), str(output_path))
    logger.info("Copied final image to: %s", output_path)
    return str(output_path.resolve())


def resolve_saved_path(path_value: Optional[str], result_dir: Path) -> Optional[Path]:
    if not path_value:
        return None

    path = Path(path_value)
    if path.is_absolute():
        return path

    if path.exists():
        return path

    candidate = result_dir / path
    if candidate.exists():
        return candidate

    return path


def original_output_name(source_path: Path) -> str:
    suffix = source_path.suffix.lower() or ".jpg"
    return f"original_image{suffix}"


def ensure_original_image(data: Dict[str, Any], result_dir: Path) -> Path:
    original = resolve_saved_path(data.get("original_image"), result_dir)
    if original is not None and original.is_file():
        return original

    source = resolve_saved_path(data.get("source_image") or data.get("image"), result_dir)
    if source is None or not source.is_file():
        raise FileNotFoundError(f"Could not find original image for result folder: {result_dir}")

    original = result_dir / original_output_name(source)
    shutil.copy2(str(source), str(original))
    data["original_image"] = str(original)
    return original


def filtered_lpd_output_name(index: int) -> str:
    return "lpd_after_preprocessing.png" if index == 0 else f"lpd_after_preprocessing_{index + 1}.png"


def load_filters_v2_module():
    filters_path = Path(__file__).with_name("filters-v2.py")
    if not filters_path.is_file():
        raise FileNotFoundError(f"Could not find filters-v2.py beside this script: {filters_path}")

    spec = importlib.util.spec_from_file_location("filters_v2", str(filters_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load filters-v2.py from: {filters_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "enhance_plate"):
        raise RuntimeError("filters-v2.py does not define enhance_plate().")
    return module


def apply_filters_v2(filters_v2, crop_path: Path, output_path: Path) -> str:
    if not crop_path.is_file():
        raise FileNotFoundError(f"Plate crop not found: {crop_path}")

    with tempfile.TemporaryDirectory() as temp_dir:
        with contextlib.redirect_stdout(io.StringIO()):
            filters_v2.enhance_plate(str(crop_path), temp_dir)
        final_stage = Path(temp_dir) / "04_sharpened.png"
        if not final_stage.is_file():
            raise FileNotFoundError(f"filters-v2.py did not create expected file: {final_stage}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(final_stage), str(output_path))

    logger.info("Saved preprocessed LPD crop: %s", output_path)
    return str(output_path.resolve())


def bbox_from_plate_record(
    plate_record: Dict[str, Any],
    image_width: int,
    image_height: int,
) -> Optional[Tuple[int, int, int, int]]:
    bbox = plate_record.get("bbox_xyxy") or {}
    if bbox:
        x1 = int(round(float(bbox.get("x1", 0))))
        y1 = int(round(float(bbox.get("y1", 0))))
        x2 = int(round(float(bbox.get("x2", 0))))
        y2 = int(round(float(bbox.get("y2", 0))))
    else:
        dims = plate_record.get("crop_dimensions") or {}
        x1 = int(round(float(dims.get("x", 0))))
        y1 = int(round(float(dims.get("y", 0))))
        x2 = x1 + int(round(float(dims.get("width", 0))))
        y2 = y1 + int(round(float(dims.get("height", 0))))

    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(0, min(x2, image_width))
    y2 = max(0, min(y2, image_height))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def paste_filtered_lpd_on_original(
    original_path: Path,
    plate_records: List[Dict[str, Any]],
    output_path: Path,
) -> Tuple[Optional[str], int]:
    original = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
    if original is None or original.size == 0:
        raise ValueError(f"Original image is empty or unreadable: {original_path}")

    result = original.copy()
    h, w = result.shape[:2]
    pasted_count = 0

    for plate_record in plate_records:
        filtered_path = resolve_saved_path(plate_record.get("lpd_after_preprocessing"), output_path.parent)
        if filtered_path is None or not filtered_path.is_file():
            plate_record["pasted_to_original"] = False
            continue

        filtered_lpd = cv2.imread(str(filtered_path), cv2.IMREAD_COLOR)
        if filtered_lpd is None or filtered_lpd.size == 0:
            plate_record["pasted_to_original"] = False
            continue

        clamped_bbox = bbox_from_plate_record(plate_record, w, h)
        if clamped_bbox is None:
            plate_record["pasted_to_original"] = False
            continue

        x1, y1, x2, y2 = clamped_bbox
        resized_lpd = cv2.resize(filtered_lpd, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
        result[y1:y2, x1:x2] = resized_lpd
        plate_record["pasted_to_original"] = True
        pasted_count += 1

    if pasted_count == 0:
        return None, 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), result):
        raise RuntimeError(f"Failed to save image after preprocessing: {output_path}")

    logger.info("Saved image after preprocessing: %s", output_path)
    return str(output_path.resolve()), pasted_count


def process_result_folder(filters_v2, result_dir: Path) -> Dict[str, Any]:
    result_dir = result_dir.resolve()
    result_json = result_dir / "result.json"
    data = load_json(result_json)
    plate_records = data.get("lpd_crops")
    if not isinstance(plate_records, list):
        plate_records = []
        data["lpd_crops"] = plate_records

    data["pipeline_stage"] = "preprocessing_lpd_back"
    data["preprocessing_error"] = None

    try:
        original_path = ensure_original_image(data, result_dir)

        for index, plate_record in enumerate(plate_records):
            if not isinstance(plate_record, dict):
                continue

            crop_path = resolve_saved_path(
                plate_record.get("raw_lpd") or plate_record.get("plate_crop"),
                result_dir,
            )
            if crop_path is None:
                plate_record["preprocessing_error"] = "Missing raw_lpd path."
                continue

            output_path = result_dir / filtered_lpd_output_name(int(plate_record.get("index", index)))
            try:
                plate_record["lpd_after_preprocessing"] = apply_filters_v2(
                    filters_v2,
                    crop_path,
                    output_path,
                )
                plate_record["preprocessing_error"] = None
            except Exception as exc:
                plate_record["preprocessing_error"] = str(exc)
                logger.error("Failed to preprocess crop=%s error=%s", crop_path, exc)

        final_path, pasted_count = paste_filtered_lpd_on_original(
            original_path,
            plate_records,
            result_dir / "image_after_preprocessing.jpg",
        )
        data["image_after_preprocessing"] = final_path
        data["pasted_filtered_lpd_count"] = pasted_count
    except Exception as exc:
        data["preprocessing_error"] = str(exc)
        logger.error("Failed result_folder=%s error=%s", result_dir, exc)

    save_json(data, result_json)
    return data


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 2: read cropped plates and saved dimensions, apply filters-v2, "
            "paste the preprocessed plate back on the original image, and update the result folder."
        )
    )
    parser.add_argument(
        "results_path",
        nargs="?",
        default=str(default_results_root()),
        help="A preprocessing_results folder, one result folder, or one result.json file.",
    )
    parser.add_argument(
        "--final-images-dir",
        help="Folder that receives one copy of every final image.",
    )
    args = parser.parse_args()

    try:
        result_folders = collect_result_folders(Path(args.results_path))
        filters_v2 = load_filters_v2_module()
        results_path = Path(args.results_path).resolve()
        if args.final_images_dir:
            final_images_dir = Path(args.final_images_dir).resolve()
        elif results_path.is_file():
            final_images_dir = results_path.parent.parent / "final images"
        elif (results_path / "result.json").is_file():
            final_images_dir = results_path.parent / "final images"
        else:
            final_images_dir = results_path.parent / "final images"
    except Exception as exc:
        logger.error("%s", exc)
        return 1

    records = []
    for result_dir in result_folders:
        record = process_result_folder(filters_v2, result_dir)
        try:
            record["final_image_copy"] = save_final_image_copy(
                record,
                result_dir,
                final_images_dir,
            )
            record.pop("final_image_copy_error", None)
            save_json(record, result_dir / "result.json")
        except Exception as exc:
            record["final_image_copy"] = None
            record["final_image_copy_error"] = str(exc)
            save_json(record, result_dir / "result.json")
            logger.error("Failed to copy final image for folder=%s error=%s", result_dir, exc)
        records.append(record)

        print(f"folder: {result_dir}")
        print(f"plate_crops: {len(record.get('lpd_crops', []))}")
        print(f"pasted_plates: {record.get('pasted_filtered_lpd_count', 0)}")
        print(f"image_after_preprocessing: {record.get('image_after_preprocessing')}")
        print(f"final_image_copy: {record.get('final_image_copy')}")
        if record.get("preprocessing_error"):
            print(f"error: {record['preprocessing_error']}")
        if record.get("final_image_copy_error"):
            print(f"copy_error: {record['final_image_copy_error']}")
        print()

    failed_count = sum(
        1
        for item in records
        if item.get("preprocessing_error")
        or item.get("final_image_copy_error")
        or any(
            isinstance(crop, dict) and crop.get("preprocessing_error")
            for crop in item.get("lpd_crops", [])
        )
    )

    print("summary:")
    print(f"  folders_processed: {len(records)}")
    print(f"  folders_with_errors: {failed_count}")
    print(f"  pasted_plates: {sum(int(item.get('pasted_filtered_lpd_count') or 0) for item in records)}")
    print(f"  final_images_folder: {final_images_dir}")
    print(f"  final_images_copied: {sum(1 for item in records if item.get('final_image_copy'))}")
    return 1 if failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
