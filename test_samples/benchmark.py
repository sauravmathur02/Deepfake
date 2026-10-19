"""
Automated Benchmark Test Suite for Universal Fake Detect
Evaluates test samples against the running server at http://localhost:8000/predict.
Supports --limit (default: 10 per category) or --all.
"""
import argparse
import os
import time
import requests

API_URL = "http://localhost:8000/predict"
TEST_DIR = os.path.dirname(os.path.abspath(__file__))

categories = [
    ("Real Images", os.path.join(TEST_DIR, "real_images"), "Real"),
    ("Fake Images", os.path.join(TEST_DIR, "fake_images"), "Fake"),
    ("Real Videos", os.path.join(TEST_DIR, "real_videos"), "Real"),
    ("Fake Videos", os.path.join(TEST_DIR, "fake_videos"), "Fake"),
]

def check_server():
    try:
        r = requests.get("http://localhost:8000/", timeout=3)
        return r.ok
    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser(description="Deepfake Detector Benchmark Evaluation")
    parser.add_argument("--limit", type=int, default=10, help="Number of files to test per category (0 = all, default: 10)")
    parser.add_argument("--all", action="store_true", help="Test all available files in every category")
    args = parser.parse_args()

    limit = 0 if args.all else args.limit

    print("=" * 80)
    print("   DEEPFAKE DETECTOR BENCHMARK EVALUATION")
    print(f"   Sample Limit: {'ALL' if limit == 0 else f'{limit} per category'}")
    print("=" * 80)
    
    if not check_server():
        print("ERROR: Server is not running at http://localhost:8000!")
        print("Please start the server first.")
        return

    total = 0
    correct = 0
    results = []

    for cat_name, folder, expected in categories:
        if not os.path.isdir(folder):
            continue
        all_files = sorted([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])
        files = all_files if limit == 0 else all_files[:limit]
        
        print(f"\n--> Testing {cat_name} ({len(files)}/{len(all_files)} items, Expected: {expected})...")
        
        for fname in files:
            fpath = os.path.join(folder, fname)
            total += 1
            ctype = "video/mp4" if fname.endswith((".mp4", ".avi", ".mov")) else "image/png"
            t0 = time.time()
            try:
                with open(fpath, "rb") as f:
                    resp = requests.post(
                        API_URL,
                        files={"file": (fname, f, ctype)},
                        timeout=120
                    )
                elapsed = time.time() - t0
                if resp.status_code == 200:
                    data = resp.json()
                    pred = data.get("prediction", "Error")
                    score = data.get("score", 0.0)
                    faces = data.get("faces_found", 0)
                    is_correct = (pred == expected)
                    if is_correct:
                        correct += 1
                    status_mark = "[PASS]" if is_correct else "[FAIL]"
                    print(f"  {status_mark} {fname[:38]:<38} | Pred: {pred:<5} (Expected: {expected}) | Score: {score:.4f} | Faces: {faces} | {elapsed:.2f}s")
                    results.append((fname, expected, pred, score, faces, is_correct, elapsed))
                else:
                    print(f"  [ERR]  {fname} -> HTTP {resp.status_code}")
            except Exception as e:
                print(f"  [ERR]  {fname} -> {e}")

    print("\n" + "=" * 80)
    acc = (correct / total * 100) if total > 0 else 0
    print(f"SUMMARY: {correct}/{total} Correct ({acc:.1f}% Accuracy)")
    print("=" * 80)

if __name__ == "__main__":
    main()
