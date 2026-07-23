@echo off
setlocal
cd /d "%~dp0"

set "INPUT_PATH=images"
set "RESULTS_ROOT=images\preprocessing_results"
set "PORTAL_URL=http://127.0.0.1:5000/"

if not "%~1"=="" set "INPUT_PATH=%~1"
if not "%~2"=="" set "RESULTS_ROOT=%~2"

set "PORTAL_PYTHON_EXE="
set "PORTAL_PYTHON_ARGS="
set "PROCESSING_PYTHON_EXE="

if exist "%USERPROFILE%\anaconda3\python.exe" set "PORTAL_PYTHON_EXE=%USERPROFILE%\anaconda3\python.exe"
if not defined PORTAL_PYTHON_EXE if exist "%USERPROFILE%\miniconda3\python.exe" set "PORTAL_PYTHON_EXE=%USERPROFILE%\miniconda3\python.exe"
if not defined PORTAL_PYTHON_EXE if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" set "PORTAL_PYTHON_EXE=%CONDA_PREFIX%\python.exe"
if not defined PORTAL_PYTHON_EXE if exist "C:\ProgramData\anaconda3\python.exe" set "PORTAL_PYTHON_EXE=C:\ProgramData\anaconda3\python.exe"
if not defined PORTAL_PYTHON_EXE if exist "C:\ProgramData\miniconda3\python.exe" set "PORTAL_PYTHON_EXE=C:\ProgramData\miniconda3\python.exe"

if not defined PORTAL_PYTHON_EXE (
    py -3 --version >nul 2>&1
    if not errorlevel 1 (
        set "PORTAL_PYTHON_EXE=py"
        set "PORTAL_PYTHON_ARGS=-3"
    )
)

if not defined PORTAL_PYTHON_EXE (
    python --version >nul 2>&1
    if not errorlevel 1 set "PORTAL_PYTHON_EXE=python"
)

if not defined PORTAL_PYTHON_EXE (
    echo ERROR: Python 3 was not found.
    echo Install Python 3 or Conda, then run this file again.
    goto :failed
)

"%PORTAL_PYTHON_EXE%" %PORTAL_PYTHON_ARGS% -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo ERROR: The portal Python is missing Flask.
    echo Python: %PORTAL_PYTHON_EXE% %PORTAL_PYTHON_ARGS%
    goto :failed
)

if exist "%USERPROFILE%\anaconda3\envs\yolo\python.exe" set "PROCESSING_PYTHON_EXE=%USERPROFILE%\anaconda3\envs\yolo\python.exe"
if not defined PROCESSING_PYTHON_EXE if exist "%USERPROFILE%\miniconda3\envs\yolo\python.exe" set "PROCESSING_PYTHON_EXE=%USERPROFILE%\miniconda3\envs\yolo\python.exe"
if not defined PROCESSING_PYTHON_EXE if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" set "PROCESSING_PYTHON_EXE=%CONDA_PREFIX%\python.exe"
if not defined PROCESSING_PYTHON_EXE set "PROCESSING_PYTHON_EXE=%PORTAL_PYTHON_EXE%"

"%PROCESSING_PYTHON_EXE%" -c "import cv2, ultralytics, openvino" >nul 2>&1
if errorlevel 1 (
    echo ERROR: The processing Python is missing cv2, ultralytics, or openvino.
    echo Python: %PROCESSING_PYTHON_EXE%
    goto :failed
)

if not exist "%INPUT_PATH%" (
    echo ERROR: Input path was not found: %INPUT_PATH%
    goto :failed
)

echo ============================================================
echo LPD pipeline
echo Processing Python: %PROCESSING_PYTHON_EXE%
echo Portal Python:     %PORTAL_PYTHON_EXE% %PORTAL_PYTHON_ARGS%
echo Input:   %INPUT_PATH%
echo Results: %RESULTS_ROOT%
echo ============================================================
echo.

echo [1/3] Detecting and cropping license plates...
"%PROCESSING_PYTHON_EXE%" "plate-cropper-for-preprocessing.py" "%INPUT_PATH%" --output-root "%RESULTS_ROOT%"
if errorlevel 1 (
    echo.
    echo ERROR: Plate detection and cropping failed.
    goto :failed
)

echo.
echo [2/3] Preprocessing the LPD crops and pasting them back...
"%PROCESSING_PYTHON_EXE%" "preprocessing-lpd-back.py" "%RESULTS_ROOT%"
if errorlevel 1 (
    echo.
    echo ERROR: LPD preprocessing or paste-back failed.
    goto :failed
)

echo.
echo [3/3] Starting the results portal...
echo URL: %PORTAL_URL%
echo Keep this window open while viewing the results.
echo Press Ctrl+C to stop the portal.
start "" powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 2; Start-Process '%PORTAL_URL%'"
"%PORTAL_PYTHON_EXE%" %PORTAL_PYTHON_ARGS% "lpd_portal.py" --results-root "%RESULTS_ROOT%" --host 127.0.0.1 --port 5000
goto :end

:failed
echo.
echo The pipeline did not complete. Review the error above.
pause
exit /b 1

:end
endlocal
