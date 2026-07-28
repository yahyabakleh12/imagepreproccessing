"""HTTP API for the license-plate preprocessing pipeline."""

import base64
import importlib.util
import json
import os
import shutil
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask


BASE_DIR = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = int(os.getenv("LPD_MAX_UPLOAD_MB", "20")) * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
PIPELINE_LOCK = threading.Lock()


def load_local_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, BASE_DIR / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cropper = load_local_module("lpd_cropper_api", "plate-cropper-for-preprocessing.py")
preprocessor = load_local_module("lpd_preprocessor_api", "preprocessing-lpd-back.py")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # These objects are expensive, so load them once and reuse them for all requests.
    previous_cwd = Path.cwd()
    try:
        os.chdir(BASE_DIR)
        app.state.plate_model = cropper.load_plate_model()
        app.state.filters = preprocessor.load_filters_v2_module()
    finally:
        os.chdir(previous_cwd)
    yield


app = FastAPI(
    title="LPD Preprocessing API",
    version="1.0.0",
    description="Upload a vehicle image and receive the image with enhanced plates pasted back.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(BASE_DIR / "dashboard.html", media_type="text/html")


def safe_suffix(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    return suffix if suffix in ALLOWED_EXTENSIONS else ".jpg"


def public_metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "has_plate": bool(record.get("has_plate")),
        "detected_plate_count": int(record.get("detected_plate_count") or 0),
        "pasted_filtered_lpd_count": int(record.get("pasted_filtered_lpd_count") or 0),
        "plate_detector_time_sec": record.get("plate_detector_time_sec"),
    }


def image_as_data_url(path_value: str | None) -> str | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_file():
        return None
    media_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def run_uploaded_image(image: UploadFile) -> tuple[Path, dict[str, Any], Path]:
    content_type = (image.content_type or "").lower()
    if content_type and not content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="The uploaded file must be an image.")

    request_dir = Path(tempfile.mkdtemp(prefix="lpd-api-"))
    input_path = request_dir / f"input{safe_suffix(image.filename)}"
    try:
        size = 0
        with input_path.open("wb") as output:
            while chunk := image.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Image is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
                    )
                output.write(chunk)

        if size == 0 or cv2.imread(str(input_path), cv2.IMREAD_COLOR) is None:
            raise HTTPException(status_code=422, detail="The image is empty or cannot be decoded.")

        results_root = request_dir / "results"
        with PIPELINE_LOCK:
            record = cropper.process_one_image(
                app.state.plate_model,
                input_path,
                results_root,
                conf=0.15,
                iou=0.3,
            )
            if record.get("error"):
                raise RuntimeError(record["error"])
            record = preprocessor.process_result_folder(
                app.state.filters,
                Path(record["result_folder"]),
            )

        if record.get("preprocessing_error"):
            raise RuntimeError(record["preprocessing_error"])

        result_value = record.get("image_after_preprocessing") or record.get("original_image")
        result_path = Path(result_value) if result_value else input_path
        if not result_path.is_file():
            raise RuntimeError("The pipeline did not produce an output image.")
        return request_dir, record, result_path
    except Exception:
        shutil.rmtree(request_dir, ignore_errors=True)
        raise


@app.post(
    "/process",
    response_class=FileResponse,
    responses={
        200: {
            "content": {"image/jpeg": {}},
            "description": "The processed image. Detection metadata is in X-LPD-Metadata.",
        }
    },
)
def process_image(image: UploadFile = File(..., description="Vehicle image to process")):
    request_dir = None
    try:
        request_dir, record, result_path = run_uploaded_image(image)
        headers = {"X-LPD-Metadata": json.dumps(public_metadata(record), separators=(",", ":"))}
        return FileResponse(
            path=result_path,
            media_type="image/jpeg",
            filename=f"{Path(image.filename or 'image').stem}_processed.jpg",
            headers=headers,
            background=BackgroundTask(shutil.rmtree, request_dir, ignore_errors=True),
        )
    except HTTPException:
        if request_dir:
            shutil.rmtree(request_dir, ignore_errors=True)
        raise
    except Exception as exc:
        if request_dir:
            shutil.rmtree(request_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Image processing failed: {exc}") from exc
    finally:
        image.file.close()


@app.post("/process/results")
def process_image_results(image: UploadFile = File(..., description="Vehicle image to process")):
    request_dir = None
    try:
        request_dir, record, final_path = run_uploaded_image(image)
        crops = [item for item in record.get("lpd_crops", []) if isinstance(item, dict)]
        first_crop = crops[0] if crops else {}
        return {
            "metadata": public_metadata(record),
            "images": {
                "original": image_as_data_url(record.get("original_image")),
                "plate_crop": image_as_data_url(
                    first_crop.get("raw_lpd") or first_crop.get("plate_crop")
                ),
                "enhanced_plate": image_as_data_url(first_crop.get("lpd_after_preprocessing")),
                "final_result": image_as_data_url(str(final_path)),
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {exc}") from exc
    finally:
        image.file.close()
        if request_dir:
            shutil.rmtree(request_dir, ignore_errors=True)
