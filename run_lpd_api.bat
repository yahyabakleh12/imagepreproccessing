@echo off
setlocal
cd /d "%~dp0"

set "LPD_API_PYTHON=python"
if exist "%USERPROFILE%\anaconda3\envs\yolo\python.exe" set "LPD_API_PYTHON=%USERPROFILE%\anaconda3\envs\yolo\python.exe"
if exist "%USERPROFILE%\miniconda3\envs\yolo\python.exe" set "LPD_API_PYTHON=%USERPROFILE%\miniconda3\envs\yolo\python.exe"

echo ============================================================
echo LPD API
echo ============================================================
echo.
echo The model may take several seconds to load.
echo The API page will open automatically at:
echo http://localhost:8000
echo.
echo Keep this window open while using the API.
echo Press Ctrl+C to stop it.
echo.

"%LPD_API_PYTHON%" -c "import fastapi, uvicorn, multipart, cv2, ultralytics, openvino" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Required Python packages are missing.
    echo.
    echo Run this command from this folder:
    echo "%LPD_API_PYTHON%" -m pip install -r requirements-api.txt
    echo.
    pause
    exit /b 1
)

start "" powershell.exe -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 8; Start-Process 'http://localhost:8000'"
"%LPD_API_PYTHON%" -m uvicorn lpd_api:app --host 0.0.0.0 --port 8000

echo.
echo The API has stopped.
pause
