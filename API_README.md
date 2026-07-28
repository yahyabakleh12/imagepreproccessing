# LPD image API

The API accepts one vehicle image, runs the existing detection and preprocessing
pipeline, and returns the processed image.

## Install and run

From this folder, using the same Python/Conda environment as the LPD backend:

```powershell
python -m pip install -r requirements-api.txt
python -m uvicorn lpd_api:app --host 0.0.0.0 --port 8000
```

On Windows you can instead start it with `run_lpd_api.bat`.

Interactive API documentation is available at:

```text
http://localhost:8000/docs
```

## Send an image

```powershell
curl.exe -X POST "http://localhost:8000/process" `
  -F "image=@1.jpg" `
  --output processed.jpg
```

The response body is the resulting JPEG. The `X-LPD-Metadata` response header
contains the plate count, pasted plate count, and detector time. If no plate is
found, the API returns the original image and reports zero plates in that header.

Health check:

```powershell
curl.exe http://localhost:8000/health
```
