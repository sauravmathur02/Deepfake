@echo off
echo ====================================================
echo   Deepfake Detector - Automated 24-Sample Benchmark
echo ====================================================
echo.
cd /d "%~dp0"
call "..\venv_gend\Scripts\activate.bat"
python benchmark.py
echo.
pause
