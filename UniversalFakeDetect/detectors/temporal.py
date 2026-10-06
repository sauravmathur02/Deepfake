"""
Temporal consistency features for video (no training, no tuning).

Takes a short run of consecutive frames and measures:
  warp_error : after motion-compensating frame t onto t+1 with optical flow,
               how much is left over. AI video tends to "boil" (texture
               changes that motion cannot explain) -> higher error.
  hf_energy  : high-frequency (sensor-noise / fine-detail) energy. Generated
               video is often over-smooth -> lower energy.
  flicker    : std of frame-to-frame mean brightness change.
  warp_jitter: variation of warp error across the clip (unstable generation).

These are raw measurements. They are NOT yet part of the verdict: thresholds
must be validated on a held-out set of real vs AI videos first.
"""
import os
import tempfile

import cv2
import numpy as np


def _read_consecutive(path: str, n: int = 16, stride: int = 2, size: int = 320):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    need = n * stride
    start = max((total - need) // 2, 0) if total > 0 else 0
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames, i = [], 0
    while len(frames) < n:
        ok, f = cap.read()
        if not ok:
            break
        if i % stride == 0:
            h, w = f.shape[:2]
            s = size / max(h, w)
            f = cv2.resize(f, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        i += 1
    cap.release()
    return frames


def temporal_features(path: str):
    g = _read_consecutive(path)
    if len(g) < 4:
        return None
    warp_errs, flickers, hf = [], [], []
    h, w = g[0].shape
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    for a, b in zip(g[:-1], g[1:]):
        flow = cv2.calcOpticalFlowFarneback(b, a, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mapx = (gx + flow[..., 0]).astype(np.float32)
        mapy = (gy + flow[..., 1]).astype(np.float32)
        warped = cv2.remap(a, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        warp_errs.append(float(np.mean(np.abs(warped.astype(np.float32) - b.astype(np.float32)))))
        flickers.append(float(b.mean() - a.mean()))
    for f in g:
        ff = f.astype(np.float32)
        hf.append(float(np.mean(np.abs(ff - cv2.GaussianBlur(ff, (0, 0), 1.5)))))
    return {
        "warp_error": float(np.mean(warp_errs)),
        "warp_jitter": float(np.std(warp_errs)),
        "flicker": float(np.std(flickers)),
        "hf_energy": float(np.mean(hf)),
        "frames": len(g),
    }


def temporal_features_bytes(data: bytes, suffix: str = ".mp4"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        p = tmp.name
    try:
        return temporal_features(p)
    finally:
        os.remove(p)
