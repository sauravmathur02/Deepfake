"""
Download sample real and deepfake video benchmark clips into:
  - test_samples/fake_videos/
  - test_samples/real_videos/
"""

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
FAKE_VID_DIR = BASE_DIR / "fake_videos"
REAL_VID_DIR = BASE_VID = BASE_DIR / "real_videos"

FAKE_VID_DIR.mkdir(exist_ok=True)
REAL_VID_DIR.mkdir(exist_ok=True)

# 5 Real videos and 5 Deepfake videos from angads24/deepfake-video
VIDEOS = [
    # Real videos
    ("real", "real_01_exit_phone_room.mp4", "https://huggingface.co/datasets/angads24/deepfake-video/resolve/main/dataset/real/01__exit_phone_room.mp4"),
    ("real", "real_02_hugging_happy.mp4", "https://huggingface.co/datasets/angads24/deepfake-video/resolve/main/dataset/real/01__hugging_happy.mp4"),
    ("real", "real_03_kitchen_pan.mp4", "https://huggingface.co/datasets/angads24/deepfake-video/resolve/main/dataset/real/01__kitchen_pan.mp4"),
    ("real", "real_04_outside_run.mp4", "https://huggingface.co/datasets/angads24/deepfake-video/resolve/main/dataset/real/01__outside_run.mp4"),
    ("real", "real_05_podium_speech.mp4", "https://huggingface.co/datasets/angads24/deepfake-video/resolve/main/dataset/real/01__podium_speech_happy.mp4"),

    # Deepfake videos
    ("fake", "fake_01_outside_talking.mp4", "https://huggingface.co/datasets/angads24/deepfake-video/resolve/main/dataset/fake/01_02__outside_talking_still_laughing__YVGY8LOK.mp4"),
    ("fake", "fake_02_walk_hall_angry.mp4", "https://huggingface.co/datasets/angads24/deepfake-video/resolve/main/dataset/fake/01_02__walk_down_hall_angry__YVGY8LOK.mp4"),
    ("fake", "fake_03_hugging_happy.mp4", "https://huggingface.co/datasets/angads24/deepfake-video/resolve/main/dataset/fake/01_03__hugging_happy__ISF9SP4G.mp4"),
    ("fake", "fake_04_podium_speech.mp4", "https://huggingface.co/datasets/angads24/deepfake-video/resolve/main/dataset/fake/01_03__podium_speech_happy__480LQD1C.mp4"),
    ("fake", "fake_05_kitchen_still.mp4", "https://huggingface.co/datasets/angads24/deepfake-video/resolve/main/dataset/fake/01_02__kitchen_still__YVGY8LOK.mp4"),
]

print("Starting video downloads via curl...")
success = 0
for v_type, v_name, v_url in VIDEOS:
    dest = (FAKE_VID_DIR if v_type == "fake" else REAL_VID_DIR) / v_name
    if dest.exists() and dest.stat().st_size > 50000:
        print(f"  [Already exists] {dest.name}")
        success += 1
        continue
    
    print(f"  Downloading [{v_type.upper()}]: {v_name} ...")
    cmd = ["curl.exe", "-sL", v_url, "-o", str(dest)]
    res = subprocess.run(cmd)
    if res.returncode == 0 and dest.exists() and dest.stat().st_size > 50000:
        print(f"    -> Done ({dest.stat().st_size / 1024 / 1024:.2f} MB)")
        success += 1
    else:
        print(f"    -> Failed or file too small: {dest.name}")

print(f"\nVideo Download Complete! ({success}/{len(VIDEOS)} ready)")
print(f"  Fake videos directory: {FAKE_VID_DIR}")
print(f"  Real videos directory: {REAL_VID_DIR}")
