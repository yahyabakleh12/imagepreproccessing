"""HTTP API for the license-plate preprocessing pipeline."""

import base64
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
from fastapi.responses import FileResponse, HTMLResponse
from starlette.background import BackgroundTask


from lpd_pipeline import PlatePipeline


MAX_UPLOAD_BYTES = int(os.getenv("LPD_MAX_UPLOAD_MB", "20")) * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
PIPELINE_LOCK = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = PlatePipeline()
    yield


app = FastAPI(
    title="LPD Preprocessing API",
    version="1.1.0",
    description="Upload a vehicle image and receive the image with enhanced plates pasted back.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LPD Image Processor</title>
  <style>
    body {
      max-width: 1200px;
      margin: 50px auto;
      padding: 20px;
      font-family: Arial, sans-serif;
      text-align: center;
    }
    .controls {
      display: flex;
      gap: 10px;
      justify-content: center;
      margin: 25px 0;
    }
    input, button {
      padding: 10px;
      font: inherit;
    }
    button {
      cursor: pointer;
    }
    button:disabled {
      cursor: wait;
    }
    #results {
      display: none;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 20px;
      margin-top: 20px;
      text-align: left;
    }
    .result-card {
      overflow: hidden;
      border: 1px solid #ddd;
      border-radius: 10px;
      background: #fff;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    .result-card img {
      display: block;
      width: 100%;
      height: 240px;
      object-fit: contain;
      background: #f5f5f5;
    }
    .result-card h2 {
      margin: 14px 16px 6px;
      font-size: 18px;
    }
    .result-card p {
      margin: 0 16px 16px;
      color: #666;
      font-size: 14px;
    }
    #message {
      min-height: 20px;
      color: #555;
    }
    .error {
      color: #b42318 !important;
    }
  </style>
</head>
<body>
  <h1>LPD Image Processor</h1>
  <div class="controls">
    <input id="image" type="file" accept="image/*">
    <button id="process" type="button">See result</button>
  </div>
  <p id="message"></p>
  <section id="results" aria-live="polite"></section>

  <script>
    const input = document.querySelector("#image");
    const button = document.querySelector("#process");
    const message = document.querySelector("#message");
    const results = document.querySelector("#results");

    function addResult(label, imageUrl, details = "") {
      if (!imageUrl) return;
      const card = document.createElement("article");
      card.className = "result-card";

      const preview = document.createElement("img");
      preview.src = imageUrl;
      preview.alt = label;
      card.appendChild(preview);

      const heading = document.createElement("h2");
      heading.textContent = label;
      card.appendChild(heading);

      if (details) {
        const description = document.createElement("p");
        description.textContent = details;
        card.appendChild(description);
      }
      results.appendChild(card);
    }

    button.addEventListener("click", async () => {
      const image = input.files[0];
      if (!image) {
        message.textContent = "Please choose an image first.";
        message.className = "error";
        return;
      }

      button.disabled = true;
      button.textContent = "Processing...";
      message.textContent = "Please wait...";
      message.className = "";
      results.replaceChildren();
      results.style.display = "none";

      const form = new FormData();
      form.append("image", image, image.name);

      try {
        const response = await fetch("/process/results", { method: "POST", body: form });
        if (!response.ok) {
          const error = await response.json().catch(() => ({}));
          throw new Error(error.detail || "Could not process the image.");
        }
        const payload = await response.json();
        addResult("Original Vehicle Image", payload.images.original);

        for (const plate of payload.plates || []) {
          const confidence = Number.isFinite(plate.confidence)
            ? `Detection confidence: ${(plate.confidence * 100).toFixed(1)}%`
            : "Detected plate crop";
          addResult(`Plate ${plate.index} - Raw Detection`, plate.raw_crop, confidence);
          for (const stage of plate.stages || []) {
            const dimensions = stage.dimensions
              ? `${stage.dimensions.width} x ${stage.dimensions.height}`
              : "";
            addResult(
              `Plate ${plate.index} - Stage ${stage.index}: ${stage.label}`,
              stage.image,
              dimensions
            );
          }
        }

        addResult("Final Vehicle Image", payload.images.final_result);
        results.style.display = "grid";
        const count = payload.metadata.detected_plate_count;
        message.textContent = count
          ? `Processing complete. ${count} plate(s) detected.`
          : "Processing complete. No plate was detected.";
      } catch (error) {
        message.textContent = error.message;
        message.className = "error";
      } finally {
        button.disabled = false;
        button.textContent = "See result";
      }
    });
  </script>
</body>
</html>
"""


def safe_suffix(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    return suffix if suffix in ALLOWED_EXTENSIONS else ".jpg"


def public_metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "has_plate": bool(record.get("detected_plate_count")),
        "detected_plate_count": int(record.get("detected_plate_count") or 0),
        "pasted_enhanced_plate_count": int(record.get("detected_plate_count") or 0),
        "plate_detector_time_sec": record.get("detector_time_seconds"),
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
            record = app.state.pipeline.process_image(input_path, results_root)

        result_value = record.get("final_image")
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
        crops = [item for item in record.get("plates", []) if isinstance(item, dict)]
        first_crop = crops[0] if crops else {}
        plate_results = []
        for plate_number, crop in enumerate(crops, start=1):
            stages = [
                {
                    "index": stage.get("index"),
                    "name": stage.get("name"),
                    "label": stage.get("label"),
                    "dimensions": stage.get("dimensions"),
                    "image": image_as_data_url(stage.get("image")),
                }
                for stage in crop.get("stages", [])
                if isinstance(stage, dict)
            ]
            plate_results.append(
                {
                    "index": plate_number,
                    "confidence": crop.get("confidence"),
                    "raw_crop": image_as_data_url(crop.get("raw_crop")),
                    "stages": stages,
                }
            )
        return {
            "metadata": public_metadata(record),
            "plates": plate_results,
            "images": {
                "original": image_as_data_url(record.get("source_image")),
                "plate_crop": image_as_data_url(first_crop.get("raw_crop")),
                "enhanced_plate": image_as_data_url(first_crop.get("enhanced_crop")),
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
