@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    py -3.11 -m venv .venv 2>nul || python -m venv .venv
    if errorlevel 1 (
        echo Python 3.11 was not found.
        pause
        exit /b 1
    )
)

".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements-dev.txt
if errorlevel 1 (
    echo Dependency installation failed.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m pytest -q -p no:cacheprovider
if errorlevel 1 (
    echo Tests failed. Build aborted.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" version.py --write

".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean --windowed --name StrokeTriage ^
    --collect-data mediapipe --collect-binaries mediapipe ^
    --hidden-import tracker --hidden-import capture ^
    --add-data "build_info.json;." ^
    main_app.py
if errorlevel 1 (
    echo Build failed.
    pause
    exit /b 1
)

copy /y destinations.json "dist\StrokeTriage\destinations.json" >nul

echo.
echo Build complete: dist\StrokeTriage\StrokeTriage.exe
echo Edit dist\StrokeTriage\destinations.json with real centres before use.
endlocal
