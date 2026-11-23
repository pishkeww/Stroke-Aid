@echo off
setlocal
cd /d "%~dp0"

if exist "dist\StrokeTriage\StrokeTriage.exe" (
    "dist\StrokeTriage\StrokeTriage.exe" %*
    goto :eof
)

if not exist ".venv\Scripts\python.exe" (
    echo First run: creating virtual environment...
    py -3.11 -m venv .venv 2>nul || python -m venv .venv
    if errorlevel 1 (
        echo Python 3.11 was not found. Install it, or use the prebuilt StrokeTriage folder.
        pause
        exit /b 1
    )
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    if exist "wheelhouse" (
        ".venv\Scripts\python.exe" -m pip install --no-index --find-links wheelhouse -r requirements.txt
    ) else (
        ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    )
    if errorlevel 1 (
        echo Dependency installation failed.
        pause
        exit /b 1
    )
)

".venv\Scripts\python.exe" main_app.py %*
if errorlevel 1 pause
endlocal
