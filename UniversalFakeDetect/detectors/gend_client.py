"""
Client for the GenD face-forensics worker.

The worker needs a different `transformers` version than the main app, so it
runs in its own virtualenv (../venv_gend). This module starts it as a child
process and talks to it over localhost. If the venv is missing, GenD is simply
disabled and the app keeps working with the other detectors.
"""
import atexit
import os
import subprocess
import sys
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, "venv_gend", "Scripts", "python.exe")
PORT = 8001
URL = f"http://127.0.0.1:{PORT}"

_proc = None


def start(timeout: float = 240.0) -> bool:
    """Launch the worker and wait until it is healthy. Returns availability."""
    global _proc
    if not os.path.exists(PY):
        print("GenD: venv_gend not found -> face detector disabled.")
        return False
    _proc = subprocess.Popen(
        [PY, "-m", "uvicorn", "gend_worker:app", "--host", "127.0.0.1", "--port", str(PORT),
         "--log-level", "warning"],
        cwd=APP_DIR,
    )
    atexit.register(stop)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if _proc.poll() is not None:
            print("GenD: worker exited early -> face detector disabled.")
            return False
        try:
            if requests.get(f"{URL}/health", timeout=2).ok:
                return True
        except requests.RequestException:
            time.sleep(1.0)
    print("GenD: worker did not start in time -> face detector disabled.")
    return False


def stop():
    global _proc
    if _proc and _proc.poll() is None:
        _proc.terminate()
    _proc = None


def score(filename: str, data: bytes, content_type: str = ""):
    """Return (gend_prob_fake or None, faces_found, face_box or None)."""
    if _proc is None or _proc.poll() is not None:
        return None, 0, None
    try:
        r = requests.post(f"{URL}/score", files={"file": (filename, data, content_type)}, timeout=120)
        j = r.json()
        return j.get("gend"), int(j.get("faces", 0)), j.get("box")
    except Exception:
        return None, 0, None
