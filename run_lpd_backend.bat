@echo off
setlocal
cd /d "%~dp0"

set "INPUT_PATH=images"
set "RESULTS_ROOT=images\preprocessing_results"
set "FINAL_IMAGES_DIR=images\final images"

if not "%~1"=="" set "INPUT_PATH=%~1"
if not "%~2"=="" set "RESULTS_ROOT=%~2"
if not "%~3"=="" set "FINAL_IMAGES_DIR=%~3"

set "PROCESSING_PYTHON_EXE="

if exist "%USERPROFILE%\anaconda3\envs\yolo\python.exe" set "PROCESSING_PYTHON_EXE=%USERPROFILE%\anaconda3\envs\yolo\python.exe"
if not defined PROCESSING_PYTHON_EXE if exist "%USERPROFILE%\miniconda3\envs\yolo\python.exe" set "PROCESSING_PYTHON_EXE=%USERPROFILE%\miniconda3\envs\yolo\python.exe"
if not defined PROCESSING_PYTHON_EXE if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" set "PROCESSING_PYTHON_EXE=%CONDA_PREFIX%\python.exe"
if not defined PROCESSING_PYTHON_EXE if exist "%USERPROFILE%\anaconda3\python.exe" set "PROCESSING_PYTHON_EXE=%USERPROFILE%\anaconda3\python.exe"
if not defined PROCESSING_PYTHON_EXE if exist "%USERPROFILE%\miniconda3\python.exe" set "PROCESSING_PYTHON_EXE=%USERPROFILE%\miniconda3\python.exe"
if not defined PROCESSING_PYTHON_EXE if exist "C:\ProgramData\anaconda3\python.exe" set "PROCESSING_PYTHON_EXE=C:\ProgramData\anaconda3\python.exe"
if not defined PROCESSING_PYTHON_EXE if exist "C:\ProgramData\miniconda3\python.exe" set "PROCESSING_PYTHON_EXE=C:\ProgramData\miniconda3\python.exe"

if not defined PROCESSING_PYTHON_EXE (
    python --version >nul 2>&1
    if not errorlevel 1 set "PROCESSING_PYTHON_EXE=python"
)

if not defined PROCESSING_PYTHON_EXE (
    echo ERROR: Python 3 was not found.
    echo Install Python 3 or Conda, then run this file again.
    goto :failed
)

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
echo LPD backend processing
echo Python:  %PROCESSING_PYTHON_EXE%
echo Input:   %INPUT_PATH%
echo Results: %RESULTS_ROOT%
echo Finals:  %FINAL_IMAGES_DIR%
echo ============================================================
echo.

echo [1/2] Detecting and cropping license plates...
"%PROCESSING_PYTHON_EXE%" "plate-cropper-for-preprocessing.py" "%INPUT_PATH%" --output-root "%RESULTS_ROOT%"
if errorlevel 1 (
    echo.
    echo ERROR: Plate detection and cropping failed.
    goto :failed
)

echo.
echo [2/2] Preprocessing the LPD crops and pasting them back...
"%PROCESSING_PYTHON_EXE%" "preprocessing-lpd-back.py" "%RESULTS_ROOT%" --final-images-dir "%FINAL_IMAGES_DIR%"
if errorlevel 1 (
    echo.
    echo ERROR: LPD preprocessing or paste-back failed.
    goto :failed
)

echo.
echo Backend processing completed successfully.
echo Results: %RESULTS_ROOT%
echo Final images: %FINAL_IMAGES_DIR%
exit /b 0

:failed
echo.
echo Backend processing did not complete. Review the error above.
pause
exit /b 1
