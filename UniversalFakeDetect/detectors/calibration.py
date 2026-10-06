"""
Feedback-driven calibration.

Labeled samples (detector scores + ground truth derived from the user's
Correct/Wrong feedback) are stored in data/feedback.jsonl. A tiny logistic
regression over [logit(commfor), logit(ufd)] is fitted on them and saved to
data/calibration.json; app.py then uses it instead of the hand-set weights.
"""
import json
import os
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(HERE, "data")
FEEDBACK_PATH = os.path.join(DATA_DIR, "feedback.jsonl")
CALIB_PATH = os.path.join(DATA_DIR, "calibration.json")
MIN_SAMPLES = 10


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def features(commfor, bfree):
    return np.stack([_logit(commfor), _logit(bfree)], axis=-1)


def load():
    # Disabled by default: a calibration fitted on a few test images overfits
    # and silently overrides the model. Opt in with DEEPFAKE_USE_CALIBRATION=1.
    if os.environ.get("DEEPFAKE_USE_CALIBRATION") != "1":
        return None
    if os.path.exists(CALIB_PATH):
        with open(CALIB_PATH) as f:
            return json.load(f)
    return None


def apply(cal, commfor, bfree):
    """Return calibrated P(fake) or None if no calibration is available."""
    if not cal:
        return None
    z = float(features(commfor, bfree) @ np.array(cal["w"]) + cal["b"])
    return float(1 / (1 + np.exp(-z)))


def add_samples(samples):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(FEEDBACK_PATH, "a") as f:
        for s in samples:
            f.write(json.dumps({"commfor": s["commfor"], "bfree": s["bfree"],
                                "is_fake": bool(s["is_fake"])}) + "\n")


def _all_samples():
    if not os.path.exists(FEEDBACK_PATH):
        return []
    with open(FEEDBACK_PATH) as f:
        return [json.loads(l) for l in f if l.strip()]


def fit():
    rows = _all_samples()
    y = np.array([r["is_fake"] for r in rows], dtype=np.float64)
    if len(rows) < MIN_SAMPLES or y.min() == y.max():
        return {"ok": False, "n": len(rows),
                "error": f"Need at least {MIN_SAMPLES} labeled samples containing both real and fake."}
    X = features([r["commfor"] for r in rows], [r.get("bfree", r.get("ufd", 0.5)) for r in rows])
    w, b = np.zeros(2), 0.0
    cw = {1: 0.5 * len(y) / y.sum(), 0: 0.5 * len(y) / (len(y) - y.sum())}  # balance classes
    sw = np.where(y == 1, cw[1], cw[0])
    for _ in range(4000):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        g = (p - y) * sw
        w -= 0.05 * (X.T @ g / len(y) + 0.01 * w)
        b -= 0.05 * g.mean()
    p = 1 / (1 + np.exp(-(X @ w + b)))
    acc = float(((p > 0.5) == (y == 1)).mean())
    old = float((((0.5 * np.array([r["commfor"] for r in rows]) +
                   0.5 * np.array([r.get("bfree", r.get("ufd", 0.5)) for r in rows])) > 0.45) == (y == 1)).mean())
    cal = {"w": w.tolist(), "b": float(b), "n": len(rows)}
    with open(CALIB_PATH, "w") as f:
        json.dump(cal, f)
    return {"ok": True, "n": len(rows), "accuracy_before": old, "accuracy_after": acc}


if __name__ == "__main__":
    result = fit()
    print(json.dumps(result, indent=2))
