@echo off
setlocal
cd /d "%~dp0"

set "RESULTS_ROOT=images\preprocessing_results"
set "PORTAL_HOST=127.0.0.1"
set "PORTAL_PORT=5000"

if not "%~1"=="" set "RESULTS_ROOT=%~1"
if not "%~2"=="" set "PORTAL_HOST=%~2"
if not "%~3"=="" set "PORTAL_PORT=%~3"

set "PORTAL_PYTHON_EXE="
set "PORTAL_PYTHON_ARGS="

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

if not exist "%RESULTS_ROOT%" (
    echo ERROR: Results folder was not found: %RESULTS_ROOT%
    echo Run run_lpd_backend.bat first, or pass a results folder as argument 1.
    goto :failed
)

set "PORTAL_URL=http://%PORTAL_HOST%:%PORTAL_PORT%/"

echo ============================================================
echo LPD results portal
echo Python:  %PORTAL_PYTHON_EXE% %PORTAL_PYTHON_ARGS%
echo Results: %RESULTS_ROOT%
echo URL:     %PORTAL_URL%
echo ============================================================
echo.
echo Keep this window open while viewing the results.
echo Press Ctrl+C to stop the portal.

start "" powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 2; Start-Process '%PORTAL_URL%'"
"%PORTAL_PYTHON_EXE%" %PORTAL_PYTHON_ARGS% "lpd_portal.py" --results-root "%RESULTS_ROOT%" --host "%PORTAL_HOST%" --port "%PORTAL_PORT%"
exit /b %errorlevel%

:failed
echo.
pause
exit /b 1

