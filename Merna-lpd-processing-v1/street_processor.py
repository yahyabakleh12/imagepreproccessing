# street_processor.py
import os
import re
import json
import base64
import binascii
import datetime as dt
import time
import math
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import box as shapely_box
from yolo_client import YoloRemoteClient, YoloServerUnavailable
from plate_runtime import has_plate as has_plate_worker
from plate_recognition import license_plate_recognition, save_plate_recognition_response
from handel_trigger import (handle_ocr_trigger,handle_omc_trigger,handle_exit_car,handle_similarity_trigger, pm)
from check_similarity_perspective import check_similarity_perspective
from logger import get_logger
from ticket_sender import send_ticket_to_db
from contextlib import contextmanager
# ---------------- Logging & Paths ----------------
logger = get_logger("street_processor", "street_processor.log")


def _log_step_timing(step_name: str, start_time: float, *, ip: str | None = None, zone: str | None = None,
                     area: str | None = None, spot: str | None = None):
    """Uniform timing log helper to surface hidden latency."""
    duration = time.perf_counter() - start_time
    logger.info(
        "STEP_TIMING name=%s took=%.3f sec ip=%s zone=%s area=%s spot=%s",
        step_name,
        duration,
        ip or "n/a",
        zone or "n/a",
        area or "n/a",
        spot or "n/a",
    )
    return duration


@contextmanager
def _time_step(step_name: str, *, ip: str | None = None, zone: str | None = None,
               area: str | None = None, spot: str | None = None):
    """Context manager for nested step timing with uniform logging."""
    _start = time.perf_counter()
    try:
        yield
    finally:
        _log_step_timing(step_name, _start, ip=ip, zone=zone, area=area, spot=spot)

# Root for saving outputs (folders + images + payload.json)
TRIGGERS_ROOT = Path(os.getenv("TRIGGERS_ROOT", "logs_by_ip")).resolve()
TRIGGERS_ROOT.mkdir(parents=True, exist_ok=True)

# ---------------- Model ----------------
DETECT_MODEL_NAME = "best11_int8_openvino_model"
DETECT_MODEL_PATH = Path("models/best11_int8_openvino_model/")

ov_model = None
try:
    ov_model = YOLO(str(DETECT_MODEL_PATH), task="detect")
    logger.info(
        "YOLO (OpenVINO) detect model loaded successfully choice=%s task=%s path=%s",
        DETECT_MODEL_NAME,
        "detect",
        DETECT_MODEL_PATH,
    )
except Exception as e:
    logger.error(
        "Failed to load YOLO (OpenVINO) detect model choice=%s task=%s path=%s error=%s",
        DETECT_MODEL_NAME,
        "detect",
        DETECT_MODEL_PATH,
        e,
    )

segmentation_client = YoloRemoteClient()

# ---------------- Resize config ----------------
# Maximum image side length used for YOLO (can be changed via env without modifying code)
YOLO_MAX_SIDE = int(os.getenv("YOLO_MAX_SIDE", "1280"))

# Direct parameter:
# True  -> delete snapshot_annotated, snapshot_annotated_perspective, and crop_forward on ignore_similarity.
# False -> keep them.
DELETE_IGNORE_SIMILARITY_SNAPSHOTS = True

# ---------------- Vehicle class IDs (car / truck / bus) ----------------
CAR_CLASS_ID = 2
TRUCK_CLASS_ID = 7
BUS_CLASS_ID = 5


# ---------------- Helpers ----------------
def safe_slug(s: str, default: str = "unknown") -> str:
    """Sanitize folder/file names to avoid illegal characters across OSes."""
    if s is None:
        return default
    s = str(s).strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9._-]", "_", s)
    return s or default


def decode_base64_image(data: str) -> bytes | None:
    """Decode Base64 image string safely."""
    if not data:
        return None
    if data.startswith("data:"):
        data = data.split(",", 1)[-1]
    try:
        return base64.b64decode(data, validate=True)
    except binascii.Error:
        pad = (-len(data)) % 4
        if pad:
            data += "=" * pad
        try:
            return base64.b64decode(data)
        except Exception:
            return None


def _parse_dt(s: str | None) -> dt.datetime | None:
    """
    Parse 'YYYY-MM-DD HH:MM:SS' or ISO-like strings safely.
    Uses datetime.fromisoformat when possible. :contentReference[oaicite:1]{index=1}
    """
    if not s:
        return None
    s = str(s).strip()
    if not s:
        return None

    # handle ISO 'T' separator and optional trailing 'Z'
    s = s.replace("T", " ")
    if s.endswith("Z"):
        s = s[:-1]

    try:
        return dt.datetime.fromisoformat(s)  # supports 'YYYY-MM-DD HH:MM:SS' too :contentReference[oaicite:2]{index=2}
    except Exception:
        try:
            return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None


def should_ignore_in_false_trigger(pm, ip: str, zone: str, area: str, spot: str, trigger_time: str | None,
                                  threshold_sec: float = 60.0) -> tuple[bool, float | None]:
    """
    Ignore IN trigger if pm has now.time_out and it's within threshold seconds from current trigger time.
    Uses timedelta.total_seconds(). :contentReference[oaicite:3]{index=3}
    """
    key = pm.format_parking_area_spot(area, spot)
    now_state = pm.get(ip, zone, key, "now")
    if not now_state:
        return False, None

    last_out_str = getattr(now_state, "time_out", None)
    if not last_out_str:
        return False, None

    t_trigger = _parse_dt(trigger_time)
    t_out = _parse_dt(last_out_str)
    if not t_trigger or not t_out:
        return False, None

    diff_sec = abs((t_trigger - t_out).total_seconds())  # :contentReference[oaicite:4]{index=4}
    return (diff_sec < float(threshold_sec)), diff_sec


def _extract_pm_keys(payload: dict, remote_ip: str | None = None):
    """Return camera_ip, zone_name, parking_area, padded_spot, and combined parking_area_spot key."""
    parking_area = str(payload.get("parking_area") or "unknown")
    zone_name = str(payload.get("zone_name") or "unknown")
    camera_ip = str(payload.get("camera_ip") or remote_ip or "unknown")
    spot_raw = payload.get("index_number") or payload.get("index") or "000"
    spot = str(spot_raw).zfill(3) if str(spot_raw).isdigit() else str(spot_raw)
    parking_area_spot = pm.format_parking_area_spot(parking_area, spot)
    return camera_ip, zone_name, parking_area, spot, parking_area_spot


def order_polygon_from_four_points(payload: dict) -> np.ndarray | None:
    """Given 4 (x,y) keys in payload, return a properly ordered 4-point polygon (int32), or None."""
    coords = [
        [payload.get("coordinate_x1"), payload.get("coordinate_y1")],
        [payload.get("coordinate_x2"), payload.get("coordinate_y2")],
        [payload.get("coordinate_x3"), payload.get("coordinate_y3")],
        [payload.get("coordinate_x4"), payload.get("coordinate_y4")],
    ]
    coords = [c for c in coords if None not in c]
    if len(coords) != 4:
        return None
    pts = np.array(coords, dtype=np.float32)
    c = np.mean(pts, axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]
    return pts.astype(np.int32).reshape((-1, 1, 2))


def _json_default(obj):
    """Fallback converter for json.dump to handle numpy-derived types gracefully."""
    try:
        import numpy as _np
        if isinstance(obj, (_np.bool_, _np.bool8)):
            return bool(obj)
        if isinstance(obj, _np.generic):
            return obj.item()
    except Exception:
        pass
    return str(obj)


def _mask_largest_vehicle_via_yolo_server(image: np.ndarray, *, ip: str, zone: str, area: str, spot: str, label: str) -> np.ndarray | None:
    if image is None or image.size == 0:
        return None

    try:
        predict_kwargs = {"conf": 0.4, "iou": 0.3, "verbose": False, "retina_masks": True}
        yolo_start = time.perf_counter()
        results = segmentation_client.predict(image, **predict_kwargs)
        _log_step_timing(f"{label}_segmentation_yolo_server", yolo_start, ip=ip, zone=zone, area=area, spot=spot)

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return None

        best_index = None
        best_area = -1.0
        best_box = None
        h, w = image.shape[:2]

        for box_index, box in enumerate(results[0].boxes):
            try:
                cls_id = int(box.cls[0].cpu().numpy())
            except Exception:
                cls_id = -1

            if cls_id not in (CAR_CLASS_ID, TRUCK_CLASS_ID, BUS_CLASS_ID):
                continue

            try:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1_i = max(0, min(int(x1), w))
                y1_i = max(0, min(int(y1), h))
                x2_i = max(0, min(int(x2), w))
                y2_i = max(0, min(int(y2), h))
            except Exception:
                continue

            area_value = float(max(0, x2_i - x1_i) * max(0, y2_i - y1_i))
            if area_value > best_area:
                best_area = area_value
                best_index = box_index
                best_box = (x1_i, y1_i, x2_i, y2_i)

        if best_index is None or best_box is None:
            return None

        masks = getattr(results[0], "masks", None)
        mask_data = getattr(masks, "data", None) if masks is not None else None
        if mask_data is not None and best_index < len(mask_data):
            selected_mask = mask_data[best_index]
            if hasattr(selected_mask, "cpu"):
                selected_mask = selected_mask.cpu().numpy()
            else:
                selected_mask = np.asarray(selected_mask)

            if selected_mask.size > 0:
                selected_mask = selected_mask.astype(np.float32)
                if selected_mask.shape[:2] != image.shape[:2]:
                    selected_mask = cv2.resize(selected_mask, (w, h), interpolation=cv2.INTER_LINEAR)

                keep_mask = selected_mask > 0.5
                if np.any(keep_mask):
                    masked_image = np.zeros_like(image)
                    masked_image[keep_mask] = image[keep_mask]
                    return masked_image

        x1_i, y1_i, x2_i, y2_i = best_box
        bbox_mask = np.zeros(image.shape[:2], dtype=bool)
        bbox_mask[y1_i:y2_i, x1_i:x2_i] = True
        masked_image = np.zeros_like(image)
        masked_image[bbox_mask] = image[bbox_mask]
        return masked_image
    except YoloServerUnavailable as e:
        logger.critical("YOLO segmentation server unavailable for %s ip=%s zone=%s area=%s spot=%s error=%s", label, ip, zone, area, spot, e)
        raise
    except Exception as e:
        logger.warning("Failed YOLO segmentation mask for %s ip=%s zone=%s area=%s spot=%s error=%s", label, ip, zone, area, spot, e)
        return None


def _mask_saved_image_via_yolo_server(path_value: str | Path | None, *, label: str, ip: str, zone: str, area: str, spot: str) -> str | None:
    if not path_value:
        return None

    path = Path(path_value)
    if not path.is_file():
        return None

    image = cv2.imread(str(path))
    if image is None or image.size == 0:
        return None

    masked_image = _mask_largest_vehicle_via_yolo_server(
        image,
        ip=ip,
        zone=zone,
        area=area,
        spot=spot,
        label=label,
    )
    if masked_image is None or masked_image.size == 0:
        return None

    write_start = time.perf_counter()
    if cv2.imwrite(str(path), masked_image):
        _log_step_timing(f"{label}_masked_write", write_start, ip=ip, zone=zone, area=area, spot=spot)
        return str(path)
    return None


def _mask_saved_forward_crop_via_yolo_server(forward_crop_path: str | None, *, ip: str, zone: str, area: str, spot: str) -> str | None:
    if not forward_crop_path:
        return None
    path = Path(forward_crop_path)
    if path.name != "crop_forward.jpg":
        return None
    return _mask_saved_image_via_yolo_server(
        path,
        label="crop_forward",
        ip=ip,
        zone=zone,
        area=area,
        spot=spot,
    )


def _recognize_forward_crop(forward_snapshot_path: str | None, *, ip: str, zone: str, area: str, spot: str):
    if not forward_snapshot_path:
        return None, None

    path = Path(forward_snapshot_path)
    if path.name != "crop_forward.jpg" or not path.is_file():
        return None, None

    plate_start = time.perf_counter()
    has_plate_bbox = _has_plate_with_lpd_file_fallback(
        str(path),
        lpd_name="forward_lpd.jpg",
        ip=ip,
        zone=zone,
        area=area,
        spot=spot,
        source="crop_forward",
    )
    plate_duration = time.perf_counter() - plate_start
    logger.info(
        "Forward crop plate worker result has_plate=%s ip=%s zone=%s area=%s spot=%s path=%s took=%.3f sec",
        has_plate_bbox,
        ip,
        zone,
        area,
        spot,
        path,
        plate_duration,
    )
    if not has_plate_bbox:
        return "OMC", None

    ocr_start = time.perf_counter()
    status, plate = license_plate_recognition(
        str(path),
        response_filename="forward_plate_recognition_response.json",
        lpd_output_name="lpd.jpg",
        lpd_folder_output_name="forward_lpd.jpg",
    )
    ocr_duration = time.perf_counter() - ocr_start
    logger.info(
        "Forward crop OCR result status=%s ip=%s zone=%s area=%s spot=%s path=%s took=%.3f sec",
        status,
        ip,
        zone,
        area,
        spot,
        path,
        ocr_duration,
    )
    return status, plate


def _has_plate_with_lpd_file_fallback(
    image_path: str,
    *,
    lpd_name: str,
    ip: str,
    zone: str,
    area: str,
    spot: str,
    source: str,
) -> bool:
    plate_worker_timeout = float(os.getenv("PLATE_WORKER_TIMEOUT", "8"))
    has_plate_bbox = has_plate_worker(image_path, timeout=plate_worker_timeout)
    if has_plate_bbox:
        return True

    lpd_path = Path(image_path).with_name("lpd") / lpd_name
    fallback_wait = float(os.getenv("LPD_FILE_FALLBACK_WAIT", "2"))
    wait_start = time.perf_counter()
    while True:
        if lpd_path.is_file():
            logger.warning(
                "Plate worker returned False but LPD crop exists; treating as has_plate=True source=%s ip=%s zone=%s area=%s spot=%s lpd=%s",
                source,
                ip,
                zone,
                area,
                spot,
                lpd_path,
            )
            return True

        if time.perf_counter() - wait_start >= fallback_wait:
            break
        time.sleep(0.05)

    return False


def _delete_path_if_exists(path_value: str | Path | None, *, label: str, ip: str, zone: str, area: str, spot: str) -> bool:
    if not path_value:
        return False
    try:
        path = Path(path_value)
        if path.exists() and path.is_file():
            path.unlink()
            logger.info(
                "Deleted %s for ignore_similarity ip=%s zone=%s area=%s spot=%s path=%s",
                label,
                ip,
                zone,
                area,
                spot,
                path,
            )
            return True
    except Exception as e:
        logger.warning(
            "Failed deleting %s for ignore_similarity ip=%s zone=%s area=%s spot=%s path=%s error=%s",
            label,
            ip,
            zone,
            area,
            spot,
            path_value,
            e,
        )
    return False


def _is_valid_base64_image(s: str) -> bool:
    if not isinstance(s, str):
        return False

    s = s.strip()

    if not s or s == "_":
        return False

    if s.startswith("data:"):
        s = s.split(",", 1)[-1]

    try:
        base64.b64decode(s, validate=True)
        return True
    except Exception:
        return False

def _normalize_out_payload_keys(payload: dict) -> dict:
    """For OUT triggers, swap current and afterward keys cleanly:
    - snapshot_afterward -> snapshot
    - snapshot -> snapshot_forward
    Then remove *_afterward keys.
    """
    if not isinstance(payload, dict):
        return payload

    try:
        if int(payload.get("occupancy")) != 0:
            return payload
    except Exception:
        return payload
        
    snap = payload.get("snapshot_afterward")

    if not _is_valid_base64_image(snap):
        return payload

    # Save original values first
    original_snapshot = payload.get("snapshot")
    original_x1 = payload.get("vehicle_frame_x1")
    original_y1 = payload.get("vehicle_frame_y1")
    original_x2 = payload.get("vehicle_frame_x2")
    original_y2 = payload.get("vehicle_frame_y2")

    afterward_snapshot = payload.get("snapshot_afterward")
    afterward_x1 = payload.get("vehicle_frame_x1_afterward")
    afterward_y1 = payload.get("vehicle_frame_y1_afterward")
    afterward_x2 = payload.get("vehicle_frame_x2_afterward")
    afterward_y2 = payload.get("vehicle_frame_y2_afterward")

    # Use afterward values for processing
    if afterward_snapshot is not None:
        payload["snapshot"] = afterward_snapshot
    if afterward_x1 is not None:
        payload["vehicle_frame_x1"] = afterward_x1
    if afterward_y1 is not None:
        payload["vehicle_frame_y1"] = afterward_y1
    if afterward_x2 is not None:
        payload["vehicle_frame_x2"] = afterward_x2
    if afterward_y2 is not None:
        payload["vehicle_frame_y2"] = afterward_y2

    # Move original values to *_forward
    if original_snapshot is not None:
        payload["snapshot_forward"] = original_snapshot
    if original_x1 is not None:
        payload["vehicle_frame_x1_forward"] = original_x1
    if original_y1 is not None:
        payload["vehicle_frame_y1_forward"] = original_y1
    if original_x2 is not None:
        payload["vehicle_frame_x2_forward"] = original_x2
    if original_y2 is not None:
        payload["vehicle_frame_y2_forward"] = original_y2

    # Remove afterward keys to avoid duplication
    payload.pop("snapshot_afterward", None)
    payload.pop("vehicle_frame_x1_afterward", None)
    payload.pop("vehicle_frame_y1_afterward", None)
    payload.pop("vehicle_frame_x2_afterward", None)
    payload.pop("vehicle_frame_y2_afterward", None)

    return payload



def _prepare_snapshot_and_annotation(img, payload, folder):
    """Save snapshot, create annotated copy, polygon, and basic labels."""
    # snapshot_path = folder / "snapshot.jpg"
    # cv2.imwrite(str(snapshot_path), img)

    annotated = img.copy()
    polygon = order_polygon_from_four_points(payload)

    zone_name = str(payload.get("zone_name", "Unknown"))
    area = str(payload.get("parking_area", "Unknown"))
    zone_region = payload.get("zone_region") or payload.get("parking_area")
    camera_id = payload.get("camera_id")
    index = str(payload.get("index_number", "N/A"))
    cv2.putText(
        annotated,
        f"{zone_name}-{area}-{index}",
        (30, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        3.5,
        (0, 255, 0),
        10,
        cv2.LINE_AA,
    )
    annotated_path = folder / "snapshot_annotated.jpg"

    return annotated, polygon, zone_name, area, zone_region, camera_id, index, annotated_path


def _draw_vehicle_frame_from_payload(annotated, payload, *, forward: bool = False):
    """Draw vehicle frame if provided in payload."""
    suffix = "_forward" if forward else ""
    vf_x1 = payload.get(f"vehicle_frame_x1{suffix}")
    vf_y1 = payload.get(f"vehicle_frame_y1{suffix}")
    vf_x2 = payload.get(f"vehicle_frame_x2{suffix}")
    vf_y2 = payload.get(f"vehicle_frame_y2{suffix}")

    if None not in (vf_x1, vf_y1, vf_x2, vf_y2):
        try:
            x1_v = int(vf_x1)
            y1_v = int(vf_y1)
            x2_v = int(vf_x2)
            y2_v = int(vf_y2)

            # draw rectangle around vehicle frame
            cv2.rectangle(
                annotated,
                (x1_v, y1_v),
                (x2_v, y2_v),
                (0, 255, 255),  # color (B,G,R) ƒ?" yellow-ish
                4,              # thickness
            )

        except Exception as e:
            pass


def _draw_trigger_coordinates_from_payload(image, payload):
    """Draw trigger polygon coordinates on the given image when available."""
    try:
        polygon = order_polygon_from_four_points(payload)
        if polygon is not None:
            cv2.polylines(image, [polygon], True, (255, 200, 0), 3)
    except Exception:
        pass


def _save_forward_snapshot(
    payload: dict,
    file_obj: dict,
    p: Path,
    folder: Path,
    *,
    ip: str,
    zone: str,
    area: str,
    spot: str,
    save_name: str = "forward_snapshot.jpg",
    allow_file_fallback: bool = True,
) -> str | None:
    """Decode forward snapshot, draw coordinates/frame, and save it."""
    img_forward = None

    snap_forward_b64 = payload.get("snapshot_forward")
    if isinstance(snap_forward_b64, str) and snap_forward_b64.strip():
        decode_start = time.perf_counter()
        img_forward_bytes = decode_base64_image(snap_forward_b64)
        _log_step_timing("snapshot_forward_base64_decode", decode_start, ip=ip, zone=zone, area=area, spot=spot)
        if img_forward_bytes:
            imdecode_start = time.perf_counter()
            nparr_forward = np.frombuffer(img_forward_bytes, np.uint8)
            img_forward = cv2.imdecode(nparr_forward, cv2.IMREAD_COLOR)
            _log_step_timing("snapshot_forward_imdecode", imdecode_start, ip=ip, zone=zone, area=area, spot=spot)
            if img_forward is None:
                logger.error("Failed to decode base64 snapshot_forward for %s", p)

    if img_forward is None and allow_file_fallback:
        snap_forward_file = file_obj.get("snapshot_forward_file") if isinstance(file_obj, dict) else None
        if isinstance(snap_forward_file, str) and Path(snap_forward_file).exists():
            file_load_start = time.perf_counter()
            img_forward = cv2.imread(snap_forward_file)
            _log_step_timing("snapshot_forward_file_read", file_load_start, ip=ip, zone=zone, area=area, spot=spot)
            logger.info("Loaded snapshot_forward from file: %s", snap_forward_file)
        elif (p.parent / "forward_snapshot.jpg").exists():
            sibling_load_start = time.perf_counter()
            img_forward = cv2.imread(str(p.parent / "forward_snapshot.jpg"))
            _log_step_timing("snapshot_forward_sibling_read", sibling_load_start, ip=ip, zone=zone, area=area, spot=spot)
            logger.info("Loaded snapshot_forward from sibling forward_snapshot.jpg")
        elif (p.parent / "exit_forward_snapshot.jpg").exists():
            sibling_load_start = time.perf_counter()
            img_forward = cv2.imread(str(p.parent / "exit_forward_snapshot.jpg"))
            _log_step_timing("snapshot_forward_sibling_read", sibling_load_start, ip=ip, zone=zone, area=area, spot=spot)
            logger.info("Loaded snapshot_forward from sibling exit_forward_snapshot.jpg")

    if img_forward is None:
        return None

    folder.mkdir(parents=True, exist_ok=True)
    forward_snapshot = img_forward.copy()
    _draw_trigger_coordinates_from_payload(forward_snapshot, payload)
    _draw_vehicle_frame_from_payload(forward_snapshot, payload, forward=True)

    zone_name = str(payload.get("zone_name", "Unknown"))
    parking_area = str(payload.get("parking_area", "Unknown"))
    index = str(payload.get("index_number", "N/A"))
    cv2.putText(
        forward_snapshot,
        f"{zone_name}-{parking_area}-{index}",
        (30, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        3.5,
        (0, 255, 0),
        10,
        cv2.LINE_AA,
    )

    forward_snapshot_path = folder / save_name
    write_start = time.perf_counter()
    cv2.imwrite(str(forward_snapshot_path), forward_snapshot)
    _log_step_timing("forward_snapshot_write", write_start, ip=ip, zone=zone, area=area, spot=spot)
    return str(forward_snapshot_path)


def _save_forward_crop_from_frame(
    payload: dict,
    folder: Path,
    *,
    ip: str,
    zone: str,
    area: str,
    spot: str,
) -> str | None:
    """Crop forward snapshot using vehicle_frame_*_forward and save as crop_forward.jpg."""
    try:
        x1 = payload.get("vehicle_frame_x1_forward")
        y1 = payload.get("vehicle_frame_y1_forward")
        x2 = payload.get("vehicle_frame_x2_forward")
        y2 = payload.get("vehicle_frame_y2_forward")

        if None in (x1, y1, x2, y2):
            return None

        try:
            x1_i = int(x1)
            y1_i = int(y1)
            x2_i = int(x2)
            y2_i = int(y2)
        except Exception:
            return None

        # Always crop from RAW snapshot_forward (base64), not from saved annotated forward image.
        img_forward = None
        snap_forward_b64 = payload.get("snapshot_forward")
        if isinstance(snap_forward_b64, str) and snap_forward_b64.strip():
            decode_start = time.perf_counter()
            img_forward_bytes = decode_base64_image(snap_forward_b64)
            _log_step_timing("crop_forward_base64_decode", decode_start, ip=ip, zone=zone, area=area, spot=spot)
            if img_forward_bytes:
                imdecode_start = time.perf_counter()
                nparr_forward = np.frombuffer(img_forward_bytes, np.uint8)
                img_forward = cv2.imdecode(nparr_forward, cv2.IMREAD_COLOR)
                _log_step_timing("crop_forward_imdecode", imdecode_start, ip=ip, zone=zone, area=area, spot=spot)

        if img_forward is None:
            return None

        h, w = img_forward.shape[:2]
        x1_c = max(0, min(x1_i, w))
        x2_c = max(0, min(x2_i, w))
        y1_c = max(0, min(y1_i, h))
        y2_c = max(0, min(y2_i, h))

        if x2_c <= x1_c or y2_c <= y1_c:
            return None

        crop_forward = img_forward[y1_c:y2_c, x1_c:x2_c]
        if crop_forward.size == 0:
            return None

        crop_forward_path = folder / "crop_forward.jpg"
        write_start = time.perf_counter()
        cv2.imwrite(str(crop_forward_path), crop_forward)
        _log_step_timing("crop_forward_write", write_start, ip=ip, zone=zone, area=area, spot=spot)
        return str(crop_forward_path)
    except Exception as e:
        logger.warning("Failed saving crop_forward ip=%s zone=%s area=%s spot=%s error=%s", ip, zone, area, spot, e)
        return None


def build_trigger_polygon_shapely(payload: dict) -> ShapelyPolygon | None:
    """
    Build a SAFE Shapely polygon from trigger coordinates.
    Fixes invalid geometries automatically.
    """
    try:
        pts = np.array([
            [payload.get("coordinate_x1"), payload.get("coordinate_y1")],
            [payload.get("coordinate_x2"), payload.get("coordinate_y2")],
            [payload.get("coordinate_x3"), payload.get("coordinate_y3")],
            [payload.get("coordinate_x4"), payload.get("coordinate_y4")],
        ], dtype=np.float64)

        if np.any(np.isnan(pts)):
            return None

     
        center = np.mean(pts, axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        pts = pts[np.argsort(angles)]

        poly = ShapelyPolygon(pts)

        
        if not poly.is_valid:
            poly = poly.buffer(0)

        if poly.is_empty or not poly.is_valid:
            return None

        return poly

    except Exception as e:
        logger.warning("Invalid trigger polygon skipped: %s", e)
        return None


def build_bbox_polygon_shapely(x1, y1, x2, y2) -> ShapelyPolygon:
    """
    Convert bbox to Shapely polygon.
    """
    return shapely_box(float(x1), float(y1), float(x2), float(y2))


def check_polygon_intersection(trigger_poly: ShapelyPolygon,
                               bbox_poly: ShapelyPolygon):
    """
    Any geometric intersection (even 1 pixel) is considered VALID.
    NO area ratio, NO thresholds.
    """
    if not trigger_poly:
        return False, None

    if not trigger_poly.intersects(bbox_poly):
        return False, None

    intersection = trigger_poly.intersection(bbox_poly)

    if intersection.is_empty:
        return False, None

    return True, intersection


def draw_intersection_polygon_cv(image, intersection_poly, color=(0, 0, 255)):
    """
    Draw intersection polygon on OpenCV image.
    """
    try:
        if intersection_poly.geom_type == "Polygon":
            coords = np.array(
                list(intersection_poly.exterior.coords),
                dtype=np.int32
            )
            cv2.polylines(image, [coords], True, color, 4)
            cv2.fillPoly(image, [coords], (0, 0, 255))
    except Exception:
        pass


def _compute_perspective_and_similarity(img, polygon, annotated_path, folder, occ_status, ip, zone, spot, parking_area_spot, zone_name, area, index):
    """Compute perspective crop and similarity metrics."""
    entry_perspective_path = None
    exit_perspective_path = None
    perspective_is_similar = None
    perspective_mean_ssim = None
    perspective_compare_image_path = None

    try:
        # reorder points TL,TR,BR,BL
        pts = polygon.reshape(-1, 2).astype(np.float32)
        pts_sorted = pts[np.argsort(pts[:, 1])]

        top = pts_sorted[:2]
        bottom = pts_sorted[2:]

        top = top[np.argsort(top[:, 0])]
        bottom = bottom[np.argsort(bottom[:, 0])]

        ordered = np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)

        tl, tr, br, bl = ordered

        # compute real width/height
        width_top = np.linalg.norm(tr - tl)
        width_bottom = np.linalg.norm(br - bl)
        max_width = int(max(width_top, width_bottom))

        height_left = np.linalg.norm(bl - tl)
        height_right = np.linalg.norm(br - tr)
        max_height = int(max(height_left, height_right))

        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(ordered, dst)
        warp_start = time.perf_counter()
        warped = cv2.warpPerspective(img, M, (max_width, max_height))
        _log_step_timing("perspective_warp", warp_start, ip=ip, zone=zone, area=area, spot=spot)

        # SAVE perspective crop
        persp_path = annotated_path.with_name(f"{annotated_path.stem}_perspective{annotated_path.suffix}")
        write_start = time.perf_counter()
        cv2.imwrite(str(persp_path), warped)
        _log_step_timing("perspective_save", write_start, ip=ip, zone=zone, area=area, spot=spot)
        # Keep path style consistent with other stored images (relative under TRIGGERS_ROOT)
        if occ_status == "in":
            entry_perspective_path = str(persp_path)
        elif occ_status == "out":
            exit_perspective_path = str(persp_path)

        # ----------------- Perspective similarity check BEFORE YOLO -----------------
        try:
            current_now = pm.get(ip, zone, parking_area_spot, "now")
            current_last = pm.get(ip, zone, parking_area_spot, "last")

            prev_path_str = None
            if current_now:
                prev_path_str = getattr(current_now, "entry_perspective_path", None)
            elif current_last:
                prev_path_str = getattr(current_last, "exit_perspective_path", None)

            if prev_path_str:
                prev_path = Path(prev_path_str)
                if not prev_path.is_absolute():
                    prev_path = Path.cwd() / prev_path

                if prev_path.exists() and persp_path.exists():
                    img_prev = cv2.imread(str(prev_path))
                    img_curr = cv2.imread(str(persp_path))
                    if img_prev is not None and img_curr is not None:
                        compare_start = time.perf_counter()
                        is_similar, mean_ssim, vis_image = check_similarity_perspective(img_prev, img_curr)
                        compare_duration = time.perf_counter() - compare_start
                        perspective_is_similar = is_similar
                        perspective_mean_ssim = float(mean_ssim)

                        compare_name = f"compare_perspective_{dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                        compare_path = folder / compare_name
                        save_compare_start = time.perf_counter()
                        cv2.imwrite(str(compare_path), vis_image)
                        _log_step_timing("perspective_compare_save", save_compare_start, ip=ip, zone=zone, area=area, spot=spot)
                        perspective_compare_image_path = str(compare_path)
                        logger.info("Perspective similarity done zone=%s area=%s spot=%s mean=%.4f similar=%s time=%.3f s",zone_name,area,index,perspective_mean_ssim,perspective_is_similar,compare_duration)
                    
                    else:
                        logger.warning("Similarity check skipped: failed to read prev/curr images prev=%s curr=%s", prev_path, persp_path)
                else:
                    logger.info("Similarity check skipped: perspective prev or curr missing prev=%s curr=%s", prev_path, persp_path)
            else:
                logger.info("Similarity check skipped: no prev perspective path (now=%s, last=%s)", bool(current_now), bool(current_last))

        except Exception as e:
            logger.warning("Perspective similarity check failed zone=%s area=%s index=%s error=%s", zone_name, area, index, e)

    except Exception as e:
        logger.error("Perspective crop failed: %s", e)

    return entry_perspective_path, exit_perspective_path, perspective_is_similar, perspective_mean_ssim, perspective_compare_image_path



def polygon_diag_from_cv(polygon_np: np.ndarray) -> float:
    pts = polygon_np.reshape(-1, 2).astype(np.float32)
    minx, miny = float(pts[:, 0].min()), float(pts[:, 1].min())
    maxx, maxy = float(pts[:, 0].max()), float(pts[:, 1].max())
    return float(math.hypot(maxx - minx, maxy - miny))

def should_exit_by_center_shift(old_center, new_center, polygon_np, k: float = 0.10, min_px: float = 8.0):
    ox, oy = old_center
    nx, ny = new_center

    dx = abs(int(nx) - int(ox))
    dy = abs(int(ny) - int(oy))

    # user requirement: may be dx or dy -> take max
    d = max(dx, dy)

    diag = polygon_diag_from_cv(polygon_np)
    thr = max(float(min_px), float(diag) * float(k))

    return (d >= thr), d, thr, dx, dy, diag



def fast_bbox_overlap(bounds, x1, y1, x2, y2) -> bool:
    minx, miny, maxx, maxy = bounds
    return not (
        x2 < minx or x1 > maxx or
        y2 < miny or y1 > maxy
    )


def _run_yolo_and_plate_pipeline(img, payload, folder, occ_status,
                                 annotated, annotated_path, polygon,
                                 zone_name, area, index, ip, zone, spot,
                                 trigger_time,
                                 entry_perspective_path, exit_perspective_path,
                                 perspective_compare_image_path, perspective_mean_ssim,
                                 zone_region, camera_id,
                                 trigger_folder,
                                 forward_snapshot_path=None,
                                 exit_forward_snapshot_path=None,
                                 bus_detected_flag=None):
    """Run YOLO and plate/OCR pipeline, returning status flags."""
    coordinate_keys = [
        "coordinate_x1",
        "coordinate_y1",
        "coordinate_x2",
        "coordinate_y2",
        "coordinate_x3",
        "coordinate_y3",
        "coordinate_x4",
        "coordinate_y4",
    ]
    coordinates = {key: payload.get(key) for key in coordinate_keys}

    car_inside = False
    has_plate_flag = False  # This boolean is returned
    crop_saved = False
    entry_center = None
    exit_center = None
    bus_detected = False
    pipeline_start = time.perf_counter()

    if occ_status in ("in", "out") and ov_model is not None:
        # Resize image for YOLO while keeping track of scale
        orig_h, orig_w = img.shape[:2]
        max_side = max(orig_w, orig_h)
        scale = 1.0
        img_for_yolo = img

        if max_side > YOLO_MAX_SIDE:
            scale = YOLO_MAX_SIDE / float(max_side)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            img_for_yolo = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        logger.info("Running YOLO on snapshot zone=%s area=%s index=%s", zone_name, area, index)

        yolo_start = time.perf_counter()
        predict_kwargs = {"conf": 0.4, "iou": 0.3, "verbose": False}
        results = ov_model.predict(img_for_yolo, **predict_kwargs)
        yolo_duration = time.perf_counter() - yolo_start
        logger.info("YOLO finished for zone=%s area=%s index=%s in %.3f seconds", zone_name, area, index, yolo_duration)

        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            inv_scale = 1.0 / scale if scale != 0 else 1.0
            # trigger_polygon_shapely = build_trigger_polygon_shapely(payload)
            bus_detected = False
            bus_intersects_trigger = False
            bus_intersection_poly = None
             

            trigger_polygon_shapely = None
            trigger_bounds = None
            has_intersection = False

            with _time_step("yolo_bus_scan", ip=ip, zone=zone, area=area, spot=spot):
                for box in results[0].boxes:
                    try:
                        cls_id = int(box.cls[0].cpu().numpy())
                    except Exception:
                        cls_id = -1

                    if cls_id != BUS_CLASS_ID:
                        continue
                    
                    # === FIRST TIME BUS FOUND ===
                    if trigger_polygon_shapely is None:
                        trigger_polygon_shapely = build_trigger_polygon_shapely(payload)
                        if trigger_polygon_shapely is None:
                            break   
                        trigger_bounds = trigger_polygon_shapely.bounds

                    x1_r, y1_r, x2_r, y2_r = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    if conf < 0.4:
                        continue

                    # Convert back to original image size
                    x1 = x1_r * inv_scale
                    y1 = y1_r * inv_scale
                    x2 = x2_r * inv_scale
                    y2 = y2_r * inv_scale

                    bus_detected = True

                    # --- Intersection check ---
                    if trigger_polygon_shapely and trigger_bounds:

                        # FAST reject (cheap)
                        if not fast_bbox_overlap(trigger_bounds, x1, y1, x2, y2):
                            continue

                        # EXPENSIVE exact check (only if overlap possible)
                        bbox_poly = build_bbox_polygon_shapely(x1, y1, x2, y2)
                        has_intersection, intersection_poly = check_polygon_intersection(
                            trigger_polygon_shapely,
                            bbox_poly
                        )

                        if has_intersection:
                            bus_intersects_trigger = True
                            bus_intersection_poly = intersection_poly


                    # Draw bus bbox
                    cv2.rectangle(
                        annotated,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 165, 255),
                        4,
                    )

                    if has_intersection:
                        bus_intersects_trigger = True
                        bus_intersection_poly = intersection_poly
                        break

                if bus_detected and bus_intersects_trigger:
                    if bus_detected_flag is not None:
                        bus_detected_flag["value"] = True

                    # Draw intersection area
                    with _time_step("bus_draw_and_save", ip=ip, zone=zone, area=area, spot=spot):
                        draw_intersection_polygon_cv(annotated, bus_intersection_poly)
                        cv2.imwrite(str(annotated_path), annotated)

                    logger.info(
                        "Bus intersects trigger polygon -> IGNORE "
                        "ip=%s zone=%s spot=%s",
                        ip, zone, spot,
                    )

                    try:
                        with _time_step("handle_exit_car_bus_intersection", ip=ip, zone=zone, area=area, spot=spot):
                            handle_exit_car(
                                inside=False,
                                ip=ip,
                                zone=zone,
                                parking_area=area,
                                spot=spot,
                                exit_image_path=str(annotated_path),
                                forward_snapshot_path=forward_snapshot_path,
                                exit_forward_path=exit_forward_snapshot_path,
                                exit_perspective_path=exit_perspective_path,
                                perspective_compare_image_path=perspective_compare_image_path,
                                coordinates=coordinates,
                                perspective_mean_ssim=perspective_mean_ssim,
                                reason="BUS_intersection",
                                exit_time=trigger_time,
                                camera_id=camera_id,
                                portal_id=payload.get("portal_id"),
                                zone_region=zone_region,
                                trigger_type=occ_status,
                                trigger_path=trigger_folder,
                                is_intersection=True,
                            )
                    except Exception as e:
                        logger.error("handle_exit_car(bus_intersection) failed ip=%s zone=%s spot=%s error=%s", ip, zone, spot, e)

                    return car_inside, crop_saved, has_plate_flag, entry_center

            with _time_step("yolo_vehicle_scan", ip=ip, zone=zone, area=area, spot=spot):
                for box_index, box in enumerate(results[0].boxes):
                    # Filter only car / truck / bus classes
                    try:
                        cls_id = int(box.cls[0].cpu().numpy())
                    except Exception:
                        cls_id = -1

                    if cls_id not in (CAR_CLASS_ID, TRUCK_CLASS_ID, BUS_CLASS_ID):
                        continue

                    x1_r, y1_r, x2_r, y2_r = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    if conf < 0.4:
                        continue

                    # Convert back to original image size
                    x1 = x1_r * inv_scale
                    y1 = y1_r * inv_scale
                    x2 = x2_r * inv_scale
                    y2 = y2_r * inv_scale

                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    inside = cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0

                    if inside:
                        car_inside = True
                        if occ_status == "in":
                            entry_center = (int(cx), int(cy))
                        elif occ_status == "out":
                            exit_center = (int(cx), int(cy))
                        # Draw on annotated image at original resolution
                        with _time_step("yolo_draw_and_save", ip=ip, zone=zone, area=area, spot=spot):
                            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                            cv2.circle(annotated, (cx, cy), 5, (255, 0, 0), -1)
                            cv2.imwrite(str(annotated_path), annotated)

                        # For "in" triggers: crop + OCR + trigger handling
                        if occ_status == "in":
                            ignore_it, diff_sec = should_ignore_in_false_trigger(
                                pm=pm,
                                ip=ip,
                                zone=zone,
                                area=area,
                                spot=spot,
                                trigger_time=trigger_time,
                                threshold_sec=60.0,
                            )

                            if ignore_it:
                                ignore_start = time.perf_counter()
                                logger.warning(
                                    "FALSE_TRIGGER ignored: IN came %.1f sec after pm.now.time_out ip=%s zone=%s area=%s spot=%s",
                                    diff_sec if diff_sec is not None else -1.0,
                                    ip, zone, area, spot
                                )

                                # send to handle_similarity_trigger (NO project changes)
                                try:
                                    handle_similarity_trigger(
                                        ip=ip,
                                        zone=zone,
                                        parking_area=area,
                                        spot=spot,
                                        occ_status=occ_status,
                                        trigger_time=trigger_time,
                                        annotated_image_path=str(annotated_path),
                                        forward_snapshot_path=forward_snapshot_path,
                                        perspective_compare_image_path=perspective_compare_image_path,
                                        perspective_mean_ssim=perspective_mean_ssim,
                                        camera_id=camera_id,
                                        zone_region=zone_region,
                                        trigger_type=occ_status,
                                        trigger_path=trigger_folder,
                                    )
                                except Exception as e:
                                    logger.warning("handle_similarity_trigger(false_trigger) failed: %s", e)
                                _log_step_timing("false_trigger_handle_similarity", ignore_start, ip=ip, zone=zone, area=area, spot=spot)

                                # stop here: no OCR / no OMC
                                has_plate_flag = False
                                crop_saved = False
                                _log_step_timing("yolo_and_plate_pipeline", pipeline_start, ip=ip, zone=zone, area=area, spot=spot)
                                return car_inside, crop_saved, has_plate_flag, entry_center
                                # =============================================================================

                            x1_i = max(int(x1), 0)
                            y1_i = max(int(y1), 0)
                            x2_i = min(int(x2), orig_w)
                            y2_i = min(int(y2), orig_h)

                            crop = img[y1_i:y2_i, x1_i:x2_i]

                            if crop.size > 0:
                                crop_path = folder / "car_crop.jpg"
                                with _time_step("crop_save", ip=ip, zone=zone, area=area, spot=spot):
                                    cv2.imwrite(str(crop_path), crop)
                                crop_saved = True

                                try:
                                    masked_crop_path = _mask_saved_image_via_yolo_server(
                                        crop_path,
                                        label="car_crop",
                                        ip=ip,
                                        zone=zone,
                                        area=area,
                                        spot=spot,
                                    )
                                    if masked_crop_path:
                                        logger.info("Masked car crop before LPD: %s", masked_crop_path)

                                    # First: use plate worker to check if there is a plate
                                    plate_start = time.perf_counter()
                                    has_plate_bbox = _has_plate_with_lpd_file_fallback(
                                        str(crop_path),
                                        lpd_name="crop_lpd.jpg",
                                        ip=ip,
                                        zone=zone,
                                        area=area,
                                        spot=spot,
                                        source="car_crop",
                                    )
                                    plate_duration = time.perf_counter() - plate_start

                                    logger.info("Plate worker call ip=%s zone=%s spot=%s took %.3f seconds", ip, zone, spot, plate_duration)

                                    ocr_source = None
                                    status = "OMC"
                                    plate = None

                                    if has_plate_bbox:
                                        ocr_start = time.perf_counter()
                                        status, plate = license_plate_recognition(
                                            str(crop_path),
                                            response_filename="crop_plate_recognition_response.json",
                                            lpd_output_name="lpd.jpg",
                                            lpd_folder_output_name="crop_lpd.jpg",
                                        )
                                        ocr_duration = time.perf_counter() - ocr_start
                                        ocr_source = "car_crop" if status == "OCR" else None
                                        logger.info("Car crop OCR result status=%s ip=%s zone=%s spot=%s took %.3f seconds", status, ip, zone, spot, ocr_duration)
                                    else:
                                        save_plate_recognition_response(
                                            {
                                                "status": False,
                                                "message": "OCR skipped because plate worker found no plate bbox in car_crop",
                                                "source": "car_crop",
                                                "has_plate_bbox": False,
                                            },
                                            str(crop_path),
                                            response_filename="crop_plate_recognition_response.json",
                                        )
                                        logger.info("Plate worker: no plate detected in car_crop ip=%s zone=%s spot=%s -> trying crop_forward OCR", ip, zone, spot)

                                    if status != "OCR":
                                        masked_forward_crop_path = _mask_saved_forward_crop_via_yolo_server(
                                            forward_snapshot_path,
                                            ip=ip,
                                            zone=zone,
                                            area=area,
                                            spot=spot,
                                        )
                                        if masked_forward_crop_path:
                                            logger.info("Masked forward crop before OCR fallback: %s", masked_forward_crop_path)

                                        forward_status, forward_plate = _recognize_forward_crop(
                                            forward_snapshot_path,
                                            ip=ip,
                                            zone=zone,
                                            area=area,
                                            spot=spot,
                                        )
                                        if forward_status == "OCR":
                                            status = forward_status
                                            plate = forward_plate
                                            ocr_source = "crop_forward"

                                    if status == "OCR" and plate is not None:
                                        has_plate_flag = True
                                        logger.info(
                                            "OCR success source=%s ip=%s zone=%s spot=%s plate=%s-%s city=%s conf=%s",
                                            ocr_source,
                                            ip,
                                            zone,
                                            spot,
                                            plate.plate_code,
                                            plate.plate_number,
                                            plate.plate_city,
                                            plate.conf,
                                        )
                                        handle_ocr_trigger(
                                            ip=ip,
                                            zone=zone,
                                            parking_area=area,
                                            spot=spot,
                                            plate_num=plate.plate_number,
                                            plate_code=plate.plate_code,
                                            plate_city=plate.plate_city,
                                            conf=plate.conf,
                                            low_conf=plate.low_conf,
                                            low_conf_char=plate.low_conf_char,
                                            crop_image_path=str(crop_path),
                                            entry_image_path=str(annotated_path),
                                            forward_snapshot_path=forward_snapshot_path,
                                            entry_perspective_path=entry_perspective_path,
                                            perspective_compare_image_path=perspective_compare_image_path,
                                            perspective_mean_ssim=perspective_mean_ssim,
                                            coordinates=coordinates,
                                            time_in=trigger_time,
                                            camera_id=camera_id,
                                            zone_region=zone_region,
                                            trigger_type=occ_status,
                                            trigger_path=trigger_folder,
                                            portal_id=payload.get("portal_id"),
                                            ocr_source=ocr_source,
                                        )
                                    else:
                                        has_plate_flag = False
                                        logger.info("OCR returned OMC for car_crop and crop_forward ip=%s zone=%s spot=%s", ip, zone, spot)
                                        handle_omc_trigger(
                                            ip=ip,
                                            zone=zone,
                                            parking_area=area,
                                            spot=spot,
                                            crop_image_path=str(crop_path),
                                            entry_image_path=str(annotated_path),
                                            forward_snapshot_path=forward_snapshot_path,
                                            entry_perspective_path=entry_perspective_path,
                                            perspective_compare_image_path=perspective_compare_image_path,
                                            perspective_mean_ssim=perspective_mean_ssim,
                                            time_in=trigger_time,
                                            camera_id=camera_id,
                                            zone_region=zone_region,
                                            trigger_type=occ_status,
                                            trigger_path=trigger_folder,
                                        )
                                except YoloServerUnavailable:
                                    logger.critical(
                                        "YOLO segmentation server unavailable during plate pipeline zone=%s area=%s index=%s. Stopping street process.",
                                        zone_name,
                                        area,
                                        index,
                                    )
                                    raise
                                except Exception as e:
                                    logger.warning("Plate pipeline failed (non-fatal) zone=%s area=%s index=%s error=%s", zone_name, area, index, e)
                        break

        # After running YOLO, add explicit verification logs for "out"
        if occ_status == "out":
            with _time_step("out_imwrite_initial", ip=ip, zone=zone, area=area, spot=spot):
                cv2.imwrite(str(annotated_path), annotated)
            exit_time = payload.get("time")

            if car_inside:
                do_exit = False
                # ===================== NEW CENTER DEBUG BLOCK ======================
                try:
                    center_block_start = time.perf_counter()
                    # Load previous center from pm
                    parking_area_spot_key = pm.format_parking_area_spot(area, spot)
                    current_now = pm.get(ip, zone, parking_area_spot_key, "now")
                    old_center = getattr(current_now, "entry_center_position", None) if current_now else None

                    if old_center and exit_center:
                        ox, oy = old_center
                        nx, ny = exit_center

                        # Compute distance (keep it for logging)
                        with _time_step("center_distance_compute", ip=ip, zone=zone, area=area, spot=spot):
                            dist = math.hypot(nx - ox, ny - oy)

                        # ===== NEW: scale-aware decision using dx/dy relative to polygon size =====
                        try:
                            with _time_step("center_shift_check", ip=ip, zone=zone, area=area, spot=spot):
                                do_exit, d, thr, dx, dy, diag = should_exit_by_center_shift(
                                    old_center, exit_center, polygon, k=0.10, min_px=8.0
                                )
                        except Exception as _e:
                            do_exit, d, thr, dx, dy, diag = (True, 0, 0.0, 0, 0, 0.0)  # fail-safe: allow exit

                        logger.info(
                            "CENTER_DEBUG old=%s new=%s dist=%.2f dx=%d dy=%d d=%d thr=%.2f diag=%.2f do_exit=%s zone=%s area=%s spot=%s",
                            old_center, exit_center, dist, dx, dy, d, thr, diag, do_exit, zone, area, spot
                        )

                        # ----- Draw directly on annotated (NO debug_img, NO extra file) -----
                        with _time_step("center_debug_draw", ip=ip, zone=zone, area=area, spot=spot):
                            cv2.circle(annotated, (ox, oy), 25, (0, 255, 0), -1)   # old center
                            cv2.circle(annotated, (nx, ny), 25, (0, 0, 255), -1)   # new center
                            cv2.line(annotated, (ox, oy), (nx, ny), (255, 255, 0), 5)

                        # Optional: show threshold/decision on the image (remove if you don't want text)
                        with _time_step("center_debug_text", ip=ip, zone=zone, area=area, spot=spot):
                            cv2.putText(
                                annotated,
                                f"dx={dx} dy={dy} d={d} thr={thr:.1f} exit={do_exit}",
                                (30, 160),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                2.0,
                                (255, 255, 255),
                                6,
                                cv2.LINE_AA,
                            )
                        with _time_step("center_debug_imwrite", ip=ip, zone=zone, area=area, spot=spot):
                            cv2.imwrite(str(annotated_path), annotated)
                        logger.info("Saved center_debug image: %s", annotated_path)
                        _log_step_timing("center_debug_block", center_block_start, ip=ip, zone=zone, area=area, spot=spot)
                        
                    else:
                        logger.info("CENTER_DEBUG skipped: old_center=%s exit_center=%s", old_center, exit_center)

                except Exception as e:
                    logger.error("CENTER_DEBUG failed: %s", e)
                # ==================================================================
                if do_exit : 
              
                    logger.warning("OUT trigger because distance big between center point zone=%s area=%s index=%s",zone_name, area, index)
                    with _time_step("handle_exit_car_do_exit", ip=ip, zone=zone, area=area, spot=spot):
                        handle_exit_car(
                            inside = False,
                            ip=ip,
                            zone=zone,
                            parking_area=area,
                            spot=spot,
                            exit_image_path=str(annotated_path),
                            forward_snapshot_path=forward_snapshot_path,
                            exit_forward_path=exit_forward_snapshot_path,
                            exit_perspective_path=exit_perspective_path,
                            perspective_compare_image_path=perspective_compare_image_path,
                            coordinates=coordinates,
                            perspective_mean_ssim=perspective_mean_ssim,
                            reason="car insaide and dis big",
                            exit_time=exit_time,
                            camera_id=camera_id,
                            portal_id=payload.get("portal_id"),
                            zone_region=zone_region,
                            trigger_type=occ_status,
                            trigger_path=trigger_folder,
                        )
                else:
                    
                    logger.warning("OUT trigger but YOLO still detects a car inside polygon zone=%s area=%s index=%s", zone_name, area, index)
                    with _time_step("handle_exit_car_yolo_detects_car", ip=ip, zone=zone, area=area, spot=spot):
                        handle_exit_car(
                                inside = car_inside,
                                ip=ip,
                                zone=zone,
                                parking_area=area,
                                spot=spot,
                                exit_image_path=str(annotated_path),
                                forward_snapshot_path=forward_snapshot_path,
                                exit_forward_path=exit_forward_snapshot_path,
                                exit_perspective_path=exit_perspective_path,
                                perspective_compare_image_path=perspective_compare_image_path,
                                coordinates=coordinates,
                                perspective_mean_ssim=perspective_mean_ssim,
                                reason="YOLO_detects_car",
                                exit_time=exit_time,
                                camera_id=camera_id,
                                portal_id=payload.get("portal_id"),
                                zone_region=zone_region,
                                trigger_type=occ_status,
                                trigger_path=trigger_folder,
                            )
            else:
                logger.info("OUT trigger verified empty by YOLO zone=%s area=%s index=%s", zone_name, area, index)
                try:
                    with _time_step("handle_exit_car_verified_out", ip=ip, zone=zone, area=area, spot=spot):
                        result = handle_exit_car(
                            inside = car_inside,
                            ip=ip,
                            zone=zone,
                            parking_area=area,
                            spot=spot,
                            exit_image_path=str(annotated_path),
                            forward_snapshot_path=forward_snapshot_path,
                            exit_forward_path=exit_forward_snapshot_path,
                            exit_perspective_path=exit_perspective_path,
                            perspective_compare_image_path=perspective_compare_image_path,
                            coordinates=coordinates,
                            perspective_mean_ssim=perspective_mean_ssim,
                            reason="YOLO_VERIFIED_OUT",
                            exit_time=exit_time,
                            camera_id=camera_id,
                            portal_id=payload.get("portal_id"),
                            zone_region=zone_region,
                            trigger_type=occ_status,
                            trigger_path=trigger_folder,
                        )
                    logger.info("handle_exit_car result=%s ip=%s zone=%s spot=%s", result, ip, zone, spot)
                except Exception as e:
                    logger.error("handle_exit_car failed ip=%s zone=%s spot=%s error=%s", ip, zone, spot, e)

    _log_step_timing("yolo_and_plate_pipeline", pipeline_start, ip=ip, zone=zone, area=area, spot=spot)
    return car_inside, crop_saved, has_plate_flag, entry_center


# ---------------- Core image pipeline ----------------
def process_image(
    img: np.ndarray,
    payload: dict,
    folder: Path,
    occ_status: str,
    trigger_folder: str,
    forward_snapshot_path: str | None = None,
    exit_forward_snapshot_path: str | None = None,
    *,
    camera_ip_override: str | None = None,
    zone_override: str | None = None,
    parking_area_override: str | None = None,
    spot_override: str | None = None,
    parking_area_spot_override: str | None = None,
) -> tuple[str | None, bool, bool, str | None, str | None, bool | None, float | None, str | None]:
    try:
        step_meta = {
            "ip": str(payload.get("camera_ip") or camera_ip_override or "unknown"),
            "zone": str(payload.get("zone_name") or zone_override or "unknown"),
            "area": str(payload.get("parking_area") or parking_area_override or "unknown"),
            "spot": str(payload.get("index_number") or payload.get("index") or spot_override or "000"),
        }
        prep_start = time.perf_counter()
        annotated, polygon, zone_name, area, zone_region, camera_id, index, annotated_path = _prepare_snapshot_and_annotation(img, payload, folder)
        _log_step_timing("prepare_snapshot_and_annotation", prep_start, **step_meta)
        entry_perspective_path = None
        exit_perspective_path = None
        perspective_is_similar = None
        perspective_mean_ssim = None
        perspective_compare_image_path = None

        # Common fields for all triggers (used by in/out handlers)
        parking_area_value = parking_area_override or area
        ip = camera_ip_override or str(payload.get("camera_ip") or "unknown")
        zone = zone_override or str(payload.get("zone_name") or "unknown")
        spot = spot_override or str(payload.get("index_number", "000")).zfill(3)
        parking_area_spot = parking_area_spot_override or pm.format_parking_area_spot(parking_area_value, spot)
        trigger_time = payload.get("time")

        if polygon is None:
            cv2.imwrite(str(annotated_path), annotated)
            return str(annotated_path), False, False, None, None, None, None, None, None

        # Draw polygon & label
        cv2.polylines(annotated, [polygon], True, (255, 200, 0), 3)

        _draw_vehicle_frame_from_payload(annotated, payload)

        # _save_polygon_crop(annotated, polygon, annotated_path)

        persp_start = time.perf_counter()
        entry_perspective_path, exit_perspective_path, perspective_is_similar, perspective_mean_ssim, perspective_compare_image_path = _compute_perspective_and_similarity(
            img,
            polygon,
            annotated_path,
            folder,
            occ_status,
            ip,
            zone,
            spot,
            parking_area_spot,
            zone_name,
            area,
            index,
        )
        _log_step_timing("compute_perspective_and_similarity", persp_start, **step_meta)

        if perspective_is_similar is not None and bool(perspective_is_similar):
            logger.info(
                "Perspective similar -> skipping processing ip=%s zone=%s spot=%s mean_ssim=%s",
                ip,
                zone,
                spot,
                perspective_mean_ssim,
            )
            if not DELETE_IGNORE_SIMILARITY_SNAPSHOTS and not annotated_path.exists():
                cv2.imwrite(str(annotated_path), annotated)

            similarity_annotated_path = str(annotated_path) if annotated_path.exists() else None
            similarity_entry_perspective_path = entry_perspective_path
            similarity_exit_perspective_path = exit_perspective_path
            similarity_forward_snapshot_path = forward_snapshot_path

            if DELETE_IGNORE_SIMILARITY_SNAPSHOTS:
                _delete_path_if_exists(
                    annotated_path,
                    label="snapshot_annotated",
                    ip=ip,
                    zone=zone,
                    area=parking_area_value,
                    spot=spot,
                )
                if entry_perspective_path:
                    _delete_path_if_exists(
                        entry_perspective_path,
                        label="snapshot_annotated_perspective",
                        ip=ip,
                        zone=zone,
                        area=parking_area_value,
                        spot=spot,
                    )
                    similarity_entry_perspective_path = None
                if exit_perspective_path:
                    _delete_path_if_exists(
                        exit_perspective_path,
                        label="snapshot_annotated_perspective",
                        ip=ip,
                        zone=zone,
                        area=parking_area_value,
                        spot=spot,
                    )
                    similarity_exit_perspective_path = None
                if forward_snapshot_path and Path(forward_snapshot_path).name == "crop_forward.jpg":
                    _delete_path_if_exists(
                        forward_snapshot_path,
                        label="crop_forward",
                        ip=ip,
                        zone=zone,
                        area=parking_area_value,
                        spot=spot,
                    )
                    similarity_forward_snapshot_path = None
                similarity_annotated_path = None

            handle_similarity_trigger(
                ip=ip,
                zone=zone,
                parking_area=parking_area_value,
                spot=spot,
                occ_status=occ_status,
                trigger_time=trigger_time,
                annotated_image_path=similarity_annotated_path,
                forward_snapshot_path=similarity_forward_snapshot_path,
                exit_forward_path=exit_forward_snapshot_path,
                perspective_compare_image_path=perspective_compare_image_path,
                perspective_mean_ssim=perspective_mean_ssim,
                camera_id=camera_id,
                zone_region=zone_region,
                trigger_type=occ_status,
                trigger_path=trigger_folder,
            )
            return (
                similarity_annotated_path,
                False,
                False,
                similarity_entry_perspective_path,
                similarity_exit_perspective_path,
                perspective_is_similar,
                perspective_mean_ssim,
                perspective_compare_image_path,
                None,
            )

        bus_detected = {"value": False}
        yolo_pipeline_start = time.perf_counter()
        car_inside, crop_saved, has_plate_flag, entry_center = _run_yolo_and_plate_pipeline(
            img,
            payload,
            folder,
            occ_status,
            annotated,
            annotated_path,
            polygon,
            zone_name,
            area,
            index,
            ip,
            zone,
            spot,
            trigger_time,
            entry_perspective_path,
            exit_perspective_path,
            perspective_compare_image_path,
            perspective_mean_ssim,
            zone_region,
            camera_id,
            trigger_folder,
            forward_snapshot_path=forward_snapshot_path,
            exit_forward_snapshot_path=exit_forward_snapshot_path,
            bus_detected_flag=bus_detected,
        )
        _log_step_timing("yolo_and_plate_pipeline_wrapper", yolo_pipeline_start, **step_meta)


        # If we got here with no save for annotated, write the version without boxes
        if not annotated_path.exists():
            cv2.imwrite(str(annotated_path), annotated)


        # NEW: Trigger OMC when NO car is inside polygon
        if occ_status == "in" and not car_inside and not bus_detected["value"]:
            logger.info("NO car detected inside polygon -> sending OMC no_inside_car")

            try:
                omc_send_start = time.perf_counter()
                send_ticket_to_db(
                    status="OMC",
                    trigger_type=occ_status,
                    trigger_path=trigger_folder,
                    camera_id=camera_id,
                    camera_ip=ip,
                    zone_name=zone,
                    zone_region=zone_region,
                    spot_number=spot,
                    plate_number=None,
                    plate_code=None,
                    plate_city=None,
                    confidence=None,
                    entry_time=trigger_time,
                    exit_time=None,
                    entry_image_path=str(annotated_path),
                    exit_image_path=None,
                    forward_snapshot_path=forward_snapshot_path,
                    crop_image_path=None,
                    perspective_compare_image_path=perspective_compare_image_path,
                    perspective_mean_ssim=perspective_mean_ssim,
                    decision="no_inside_car",
                )
                _log_step_timing("send_ticket_no_inside_car", omc_send_start, **step_meta)
            except Exception as e:
                logger.error("Failed send_ticket_to_db(no_inside_car): %s", e)

        return (
            str(annotated_path),
            (car_inside and crop_saved),
            has_plate_flag,
            entry_perspective_path,
            exit_perspective_path,
            perspective_is_similar,
            perspective_mean_ssim,
            perspective_compare_image_path,
            entry_center,
        )

    except YoloServerUnavailable:
        logger.critical("YOLO segmentation server unavailable while processing image. Stopping street process.")
        raise
    except Exception as e:
        logger.error("Image processing failed: %s", e)
        return None, False, False, None, None, None, None, None, None


# ---------------- File runner (drop-in for /trigger) ----------------
async def process_trigger_file(json_path: str) -> dict:
    """
    Process a previously-saved trigger JSON exactly like the /trigger endpoint.

    Folder structure:
      logs_by_ip/<zone_name>/<camera_ip>/<parking_area>/<index>/<timestamp>_occ_<status>/
    """
    start_perf = time.perf_counter()

    p = Path(json_path)
    if not p.exists():
        logger.error("Trigger JSON does not exist: %s", p)
        duration = time.perf_counter() - start_perf
        logger.info("Trigger %s finished with error=json_not_found in %.3f seconds", p, duration)
        return {
            "ok": False,
            "error": "json_not_found",
            "json_path": str(p),
            "duration_sec": round(duration, 3),
        }

    logger.info("______________________________________________   ______________________________________________________ ")
    logger.info("Processing trigger file: %s", p)

    # Load JSON (accept raw payload or {"payload": {...}})
    try:
        json_load_start = time.perf_counter()
        with open(p, "r", encoding="utf-8") as f:
            file_obj = json.load(f)
        _log_step_timing("json_load", json_load_start)
    except Exception as e:
        logger.error("Failed to load JSON from %s error=%s", p, e)
        duration = time.perf_counter() - start_perf
        logger.info("Trigger %s finished with error=json_load_failed in %.3f seconds", p, duration)
        return {
            "ok": False,
            "error": "json_load_failed",
            "json_path": str(p),
            "duration_sec": round(duration, 3),
        }

    payload = file_obj.get("payload") if isinstance(file_obj, dict) else None
    if not isinstance(payload, dict):
        payload = file_obj if isinstance(file_obj, dict) else {}
    payload = _normalize_out_payload_keys(payload)

    # Timestamps
    now = dt.datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    base_tag = now.strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
    date_folder = TRIGGERS_ROOT / now.strftime("%Y-%m-%d")
    date_folder.mkdir(parents=True, exist_ok=True)

    # Keys for folder routing
    camera_ip_raw, zone_raw, parking_area_raw, spot_pm, parking_area_spot = _extract_pm_keys(
        payload, file_obj.get("remote_ip") if isinstance(file_obj, dict) else None
    )
    zone_name = safe_slug(zone_raw, "unknown_zone")
    cam_ip = safe_slug(camera_ip_raw, "unknown_ip")
    area = safe_slug(str(parking_area_raw), "unknown")
    index = safe_slug(spot_pm, "unknown")

    occ = payload.get("occupancy")
    try:
        occ_int = int(occ)
        occ_status = "in" if occ_int == 1 else "out" if occ_int == 0 else "unknown"
    except Exception:
        occ_status = "unknown"

    # Trigger timestamp (fallback to server time when missing).
    trigger_time = payload.get("time") or now_str

    # Build folder path
    folder_tag = f"{base_tag}_occ_{occ_status}"
    folder = date_folder / zone_name / cam_ip / area / index / folder_tag
    folder_mkdir_start = time.perf_counter()
    folder.mkdir(parents=True, exist_ok=True)
    _log_step_timing("trigger_folder_prepare", folder_mkdir_start, ip=camera_ip_raw, zone=zone_raw, area=parking_area_raw, spot=spot_pm)
    trigger_folder = str(folder.resolve())
    print(folder)
    logger.info("Trigger routing zone=%s ip=%s area=%s index=%s occ_status=%s folder=%s", zone_name, cam_ip, area, index, occ_status, folder)

    annotated_path = None
    car_inside_spot = False
    has_plate_flag = False
    entry_perspective_path = None
    exit_perspective_path = None
    perspective_is_similar = None
    perspective_mean_ssim = None
    perspective_compare_image_path = None
    entry_perspective_path = None
    exit_perspective_path = None
    entry_center = None
    forward_snapshot_path = None
    forward_crop_path = None
    forward_path_for_db = None
    exit_forward_path_for_db = None

    try:
        is_out_trigger = occ_status == "out"
        forward_file_name = "exit_forward_snapshot.jpg" if is_out_trigger else "forward_snapshot.jpg"
        if occ_status == "in":
            snapshot_forward = payload.get("snapshot_forward")
            if isinstance(snapshot_forward, str) and snapshot_forward.strip():
                forward_snapshot_path = _save_forward_snapshot(
                    payload,
                    file_obj if isinstance(file_obj, dict) else {},
                    p,
                    folder,
                    ip=camera_ip_raw,
                    zone=zone_raw,
                    area=parking_area_raw,
                    spot=spot_pm,
                    save_name=forward_file_name,
                    allow_file_fallback=True,
                )
                if forward_snapshot_path:
                    logger.info("Saved forward snapshot image: %s", forward_snapshot_path)

                forward_crop_path = _save_forward_crop_from_frame(
                    payload,
                    folder,
                    ip=camera_ip_raw,
                    zone=zone_raw,
                    area=parking_area_raw,
                    spot=spot_pm,
                )
                if forward_crop_path:
                    logger.info("Saved forward cropped image: %s", forward_crop_path)
                else:
                    logger.warning(
                        "snapshot_forward exists but crop_forward was not saved ip=%s zone=%s area=%s spot=%s",
                        camera_ip_raw,
                        zone_raw,
                        parking_area_raw,
                        spot_pm,
                    )
            if not forward_crop_path:
                if not forward_snapshot_path:
                    forward_snapshot_path = _save_forward_snapshot(
                        payload,
                        file_obj if isinstance(file_obj, dict) else {},
                        p,
                        folder,
                        ip=camera_ip_raw,
                        zone=zone_raw,
                        area=parking_area_raw,
                        spot=spot_pm,
                        save_name=forward_file_name,
                        allow_file_fallback=True,
                    )
            # IN tickets must send the cropped forward image only. Keep the full
            # forward snapshot on disk/metadata, but do not use it as the ticket
            # forward image when crop_forward.jpg could not be created.
            forward_path_for_db = forward_crop_path
        elif occ_status == "out":
            forward_snapshot_path = _save_forward_snapshot(
                payload,
                file_obj if isinstance(file_obj, dict) else {},
                p,
                folder,
                ip=camera_ip_raw,
                zone=zone_raw,
                area=parking_area_raw,
                spot=spot_pm,
                save_name=forward_file_name,
                allow_file_fallback=False,
            )
            # For OUT triggers keep only the forward snapshot with coordinates (no crop).
            exit_forward_path_for_db = forward_snapshot_path
            if forward_snapshot_path is None:
                logger.info(
                    "OUT trigger without snapshot_forward ip=%s zone=%s area=%s spot=%s",
                    camera_ip_raw,
                    zone_raw,
                    parking_area_raw,
                    spot_pm,
                )
        else:
            logger.info(
                "Skipping forward image save for non-IN trigger ip=%s zone=%s area=%s spot=%s occ_status=%s",
                camera_ip_raw,
                zone_raw,
                parking_area_raw,
                spot_pm,
                occ_status,
            )
    except Exception as e:
        logger.warning("Failed to process snapshot_forward for trigger %s error=%s", p, e)

    # Choose image source
    img = None
    snap_b64 = payload.get("snapshot")
    if isinstance(snap_b64, str) and snap_b64.strip():
        decode_start = time.perf_counter()
        img_bytes = decode_base64_image(snap_b64)
        _log_step_timing("snapshot_base64_decode", decode_start, ip=camera_ip_raw, zone=zone_raw, area=parking_area_raw, spot=spot_pm)
        if img_bytes:
            imdecode_start = time.perf_counter()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            _log_step_timing("snapshot_imdecode", imdecode_start, ip=camera_ip_raw, zone=zone_raw, area=parking_area_raw, spot=spot_pm)
            if img is None:
                logger.error("Failed to decode base64 snapshot for %s", p)

    if img is None:
        snap_file = file_obj.get("snapshot_file")
        if isinstance(snap_file, str) and Path(snap_file).exists():
            file_load_start = time.perf_counter()
            img = cv2.imread(snap_file)
            _log_step_timing("snapshot_file_read", file_load_start, ip=camera_ip_raw, zone=zone_raw, area=parking_area_raw, spot=spot_pm)
            logger.info("Loaded snapshot from file: %s", snap_file)
        elif (p.parent / "snapshot.jpg").exists():
            sibling_load_start = time.perf_counter()
            img = cv2.imread(str(p.parent / "snapshot.jpg"))
            _log_step_timing("snapshot_sibling_read", sibling_load_start, ip=camera_ip_raw, zone=zone_raw, area=parking_area_raw, spot=spot_pm)
            logger.info("Loaded snapshot from sibling snapshot.jpg")

    if img is not None:
        image_process_start = time.perf_counter()
        annotated_path, car_inside_spot, has_plate_flag, entry_perspective_path, exit_perspective_path, perspective_is_similar, perspective_mean_ssim, perspective_compare_image_path, entry_center = process_image(
            img,
            payload,
            folder,
            occ_status,
            trigger_folder,
            forward_snapshot_path=forward_path_for_db,
            exit_forward_snapshot_path=exit_forward_path_for_db,
            camera_ip_override=camera_ip_raw,
            zone_override=zone_raw,
            parking_area_override=parking_area_raw,
            spot_override=spot_pm,
            parking_area_spot_override=parking_area_spot,
        )
        _log_step_timing("process_image_total", image_process_start, ip=camera_ip_raw, zone=zone_raw, area=parking_area_raw, spot=spot_pm)
        if (
            DELETE_IGNORE_SIMILARITY_SNAPSHOTS
            and perspective_is_similar is not None
            and bool(perspective_is_similar)
            and forward_path_for_db
            and Path(forward_path_for_db).name == "crop_forward.jpg"
        ):
            forward_path_for_db = None
    else:
        logger.warning("No usable snapshot found (base64/file) for trigger %s, skipping image processing.", p)

    # Persist perspective paths in pm regardless of downstream accept/ignore outcomes.
    try:
        pm_persist_start = time.perf_counter()
        if occ_status == "in" and entry_perspective_path:
            pm.set(
                camera_ip_raw,
                zone_raw,
                parking_area_spot,
                "now",
                entry_perspective_path=entry_perspective_path,
                # time_in=trigger_time,
                process_time_in=now_str,
                entry_center_position=entry_center,
            )
        if occ_status == "out" and exit_perspective_path:
            pm.set(
                camera_ip_raw,
                zone_raw,
                parking_area_spot,
                "last",
                exit_perspective_path=exit_perspective_path,
                # time_out=trigger_time,
                process_time_out=now_str,
            )
        _log_step_timing("pm_persist", pm_persist_start, ip=camera_ip_raw, zone=zone_raw, area=parking_area_raw, spot=spot_pm)
    except Exception as e:
        logger.warning("Failed to persist perspective path in pm ip=%s zone=%s spot=%s occ=%s error=%s", camera_ip_raw, zone_raw, spot_pm, occ_status, e)

    duration = time.perf_counter() - start_perf

    # Remove snapshot from payload before storing payload.json
    if isinstance(payload, dict):
        payload.pop("snapshot", None)
        payload.pop("snapshot_forward", None)
        payload.pop("snapshot_afterward", None)
        

    rec = {
        "ok": True,
        "timestamp": now_str,
        "path": str(p),
        "bytes": p.stat().st_size if p.exists() else 0,
        "payload": payload,
        "occupancy": occ_status,
        "annotated_file": annotated_path,
        "forward_snapshot_file": forward_path_for_db,
        "forward_full_snapshot_file": forward_snapshot_path,
        "forward_crop_file": forward_crop_path,
        "exit_forward": exit_forward_path_for_db,
        "exit_forward_file": exit_forward_path_for_db,
        "car_inside_spot": car_inside_spot,
        "has_plate": has_plate_flag,
        "entry_perspective_path": entry_perspective_path,
        "exit_perspective_path": exit_perspective_path,
        "perspective_is_similar": perspective_is_similar,
        "perspective_mean_ssim": perspective_mean_ssim,
        "perspective_compare_image_path": perspective_compare_image_path,
        "entry_center_position": entry_center,
        "source_json": str(p),
        "duration_sec": round(duration, 3),
    }

    try:
        write_payload_start = time.perf_counter()
        out_path = folder / "payload.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2, default=_json_default)
        _log_step_timing("payload_write", write_payload_start, ip=camera_ip_raw, zone=zone_raw, area=parking_area_raw, spot=spot_pm)
        logger.info("Processed trigger zone=%s ip=%s area=%s index=%s folder=%s car_inside=%s has_plate=%s duration=%.3f s", zone_name, cam_ip, area, index, folder_tag, car_inside_spot, has_plate_flag, duration)
    except Exception as e:
        logger.error("Failed writing output record for %s error=%s", p, e)
        logger.info("Trigger %s finished with error on write in %.3f seconds", p, duration)


    return rec
