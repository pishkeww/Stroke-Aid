@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    py -3.11 -m venv .venv 2>nul || python -m venv .venv
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
)
if "%1"=="" (
    echo Usage:
    echo   selftest.bat run --subject H01 --condition normal --reps 3 [--source 0] [--rotate 90] [--mirror]
    echo   selftest.bat report [--save report.md]
    goto :eof
)
".venv\Scripts\python.exe" selftest.py %*
endlocal
