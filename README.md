# License Plate Preprocessing Pipeline

This project has one processing flow:

1. Accept one vehicle image or a folder of images.
2. Detect license plates with the included OpenVINO model.
3. Save every plate crop and its coordinates/dimensions.
4. Run the v1 enhancement pipeline on each crop.
5. Resize and paste each enhanced crop back into the original vehicle image.

## Setup

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The detector is already stored in
`license_plate_detector_int8_openvino_model/`.

## Command line

Process one image:

```powershell
python lpd_pipeline.py 1.jpg -o output
```

Process every supported image inside a folder (including subfolders):

```powershell
python lpd_pipeline.py images1 -o output
```

Optional detector settings:

```powershell
python lpd_pipeline.py images1 -o output --conf 0.15 --iou 0.30
```

Each image gets its own output folder containing:

- `original.<source-extension>`: unchanged copy of the source vehicle image
- `final.jpg`: vehicle image with enhanced plates pasted back
- `plate_1_raw.png`: original detected plate crop
- `plate_1_enhanced.png`: v1 enhancement result
- `result.json`: bounding box, crop dimensions, confidence, and file paths

If no plate is detected, `final.jpg` is still created as a copy of the input
and `detected_plate_count` is `0`.

## Use from another Python project

Load `PlatePipeline` once, then reuse it so the detector model is not reloaded:

```python
from lpd_pipeline import PlatePipeline

pipeline = PlatePipeline()

# One image
record = pipeline.process_image("car.jpg", "output")
print(record["final_image"])
print(record["plates"][0]["dimensions"])

# One image or a complete folder
records = pipeline.process("input_images", "output")
```

The v1 enhancement pipeline runs two stages in the default order `12`:

1. Swin2SR super-resolution selects the x2 or x4 model automatically and
   repeats when necessary until the plate reaches its target dimensions.
2. Adaptive preprocessing measures noise and blur, conditionally denoises and
   deconvolves, applies CLAHE, and conditionally sharpens.

The integration class is `v1.EnhancementPipelineV1`. It accepts OpenCV BGR or grayscale
images and performs the required RGB conversion internally.

## Optional HTTP API

```powershell
python -m uvicorn lpd_api:app --host 0.0.0.0 --port 8002
```

Open `http://localhost:8002/docs`, or send an image:

```powershell
curl.exe -X POST "http://localhost:8002/process" `
  -F "image=@1.jpg" `
  --output processed.jpg
```

`POST /process/results` returns metadata and preview images as JSON.

## Main files

- `lpd_pipeline.py`: complete reusable processing pipeline and CLI
- `v1.py`: reusable v1 enhancement pipeline composition
- `1_swin2sr.py`: Swin2SR stage
- `2_preprocess.py`: adaptive plate preprocessing stage
- `wetransfer_swin2sr_2x-bin_2026-07-28_0652/`: local x2 and x4 checkpoints
- `lpd_api.py`: optional FastAPI wrapper
- The simple browser interface is included directly in `lpd_api.py`
- `run_lpd_api.bat`: optional Windows API launcher
- `requirements.txt`: Python dependencies
