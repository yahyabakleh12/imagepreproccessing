# LPD API Test Package

This package contains only the files required to run and test the license-plate
processing API.

## Requirements

- Windows 10 or newer
- Python 3.10 or newer
- About 3 GB of free disk space for Python dependencies

## First-time setup

Open Command Prompt inside the extracted package and run:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Start the API

With the virtual environment activated:

```bat
python -m uvicorn lpd_api:app --host 127.0.0.1 --port 8002
```

Alternatively, run:

```bat
run_lpd_api.bat
```

## Test

Open the simple upload interface:

```text
http://localhost:8002
```

Open the generated API documentation:

```text
http://localhost:8002/docs
```

Health check:

```bat
curl.exe http://localhost:8002/health
```

Process an image:

```bat
curl.exe -X POST "http://localhost:8002/process" -F "image=@car.jpg" --output result.jpg
```

The first request is slower because the models are initialized. CPU processing
can take significant time because v1 runs Swin2SR enhancement. Models remain
loaded while the API process stays running.
