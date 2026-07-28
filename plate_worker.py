import cv2
import time
from pathlib import Path
from multiprocessing import Queue
from ultralytics import YOLO

from logger import get_logger

logger = get_logger("plate_worker", "plate_worker.log")


def _save_detected_plate_crop(frame, image_path: str, xyxy, output_name: str) -> str | None:
    """Save one detected plate crop beside the source car crop."""
    try:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            logger.warning("Invalid LPD bbox for image=%s bbox=%s", image_path, xyxy)
            return None

        plate_crop = frame[y1:y2, x1:x2]
        if plate_crop.size == 0:
            logger.warning("Empty LPD crop for image=%s bbox=%s", image_path, xyxy)
            return None

        output_dir = Path(image_path).with_name("lpd")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_name
        if not cv2.imwrite(str(output_path), plate_crop):
            logger.warning("Failed to save LPD crop: %s", output_path)
            return None

        logger.info("Saved LPD crop from detector: %s", output_path)
        return str(output_path)
    except Exception as e:
        logger.error("Error while saving LPD crop image=%s error=%s", image_path, e)
        return None


def _lpd_output_name_for_source(image_path: str, index: int) -> str:
    source_name = Path(image_path).stem.lower()
    if source_name == "crop_forward":
        base_name = "forward_lpd"
    elif source_name == "car_crop":
        base_name = "crop_lpd"
    else:
        base_name = "lpd"

    return f"{base_name}.jpg" if index == 0 else f"{base_name}_{index + 1}.jpg"


def _save_all_detected_plate_crops(frame, image_path: str, xyxy_list) -> list[str]:
    """Save all detected plate crops with source-specific names."""
    saved_paths = []
    for idx, xyxy in enumerate(xyxy_list):
        output_name = _lpd_output_name_for_source(image_path, idx)
        saved_path = _save_detected_plate_crop(frame, image_path, xyxy, output_name)
        if saved_path:
            saved_paths.append(saved_path)

    logger.info(
        "Saved %d/%d LPD crops from detector for image=%s",
        len(saved_paths),
        len(xyxy_list),
        image_path,
    )
    return saved_paths


def plate_worker(job_queue: Queue, result_queue: Queue):
    """
    Persistent worker that keeps the model loaded and listens for jobs.
    Each job is: (image_path, job_id)
    Result is: (job_id, has_plate: bool)

    - Loads the OpenVINO plate detector once.
    - For each image:
      * Runs YOLO to detect plates
      * Measures inference time
      * If a plate is found, draws bounding boxes on the image
        and saves an annotated copy in the same folder.
    """
    try:
        logger.info("Loading license plate detector model in plate_worker...")
        model = YOLO("models/license_plate_detector_int8_openvino_model", task="detect")
        logger.info("License plate detector model loaded in worker.")
    except Exception as e:
        logger.error("Failed to load license plate detector model in worker: %s", e)
        return

    while True:
        job = job_queue.get()
        if job == "STOP":
            logger.info("Plate worker received STOP signal, exiting.")
            break

        image_path, job_id = job
        try:
            frame = cv2.imread(image_path)
            if frame is None or frame.size == 0:
                logger.warning(
                    "Empty or unreadable image in worker job_id=%s path=%s",
                    job_id,
                    image_path,
                )
                result_queue.put((job_id, False))
                continue

            # -------------------------------
            # YOLO plate detection + timing
            # -------------------------------
            yolo_start = time.perf_counter()
            results = model(
                frame,
                conf=0.15,
                iou=0.3,
                verbose=False,
            )[0]
            yolo_duration = time.perf_counter() - yolo_start

            xyxy = results.boxes.xyxy if results.boxes is not None else []
            count = len(xyxy)
            has_plate = count >= 1

            if count > 0:
                _save_all_detected_plate_crops(frame, image_path, xyxy)

            logger.info(
                "Worker job_id=%s image=%s → detected_plates=%d has_plate=%s (YOLO time=%.3f s)",
                job_id,
                image_path,
                count,
                has_plate,
                yolo_duration,
            )
            result_queue.put((job_id, has_plate))

        except Exception as e:
            logger.error(
                "Error in plate_worker job_id=%s image=%s error=%s",
                job_id,
                image_path,
                e,
            )
            result_queue.put((job_id, False))
