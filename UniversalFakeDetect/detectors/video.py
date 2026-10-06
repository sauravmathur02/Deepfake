"""
Video helpers.

The old code looked only at the first 32 frames (~1 second). AI artifacts are
often stronger in later frames or during motion, so we sample evenly across
the whole clip.
"""
import os
import tempfile

import cv2
import numpy as np
from PIL import Image


def sample_video_frames(path: str, n_frames: int = 16) -> list:
    """Return up to n_frames PIL images spread evenly over the whole video."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = []
    if total > 0:
        for idx in np.linspace(0, max(total - 1, 0), num=min(n_frames, total)).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    if not frames:  # some containers don't report frame count / don't support seeking
        cap.release()
        cap = cv2.VideoCapture(path)
        all_frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            all_frames.append(frame)
        for idx in np.linspace(0, len(all_frames) - 1, num=min(n_frames, len(all_frames))).astype(int):
            frames.append(Image.fromarray(cv2.cvtColor(all_frames[idx], cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def sample_video_bytes(data: bytes, n_frames: int = 16, suffix: str = ".mp4") -> list:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        return sample_video_frames(tmp_path, n_frames)
    finally:
        os.remove(tmp_path)
