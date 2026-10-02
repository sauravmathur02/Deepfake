@echo off
echo ============================================
echo   Deepfake Detector - Starting Server...
echo ============================================
cd /d "%~dp0UniversalFakeDetect"
echo.
echo Installing / verifying dependencies...
pip install -r ..\requirements.txt -q
echo.
echo Starting server... Please wait for "Model ready."
echo Open your browser at: http://localhost:8000
echo Press Ctrl+C to stop.
echo.
uvicorn app:app --host 0.0.0.0 --port 8000
pause
