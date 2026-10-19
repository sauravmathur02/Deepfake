"""
Benchmark script to test all 131 files in manual_test_library against http://localhost:8000/predict.
"""
import os
import sys
import time
import json
import requests

API_URL = "http://localhost:8000/predict"
BASE = os.path.dirname(os.path.abspath(__file__))
LIB_DIR = os.path.join(BASE, "manual_test_library")

categories = [
    ("Real Images", os.path.join(LIB_DIR, "real_images"), "Real", "image/jpeg"),
    ("Fake Images", os.path.join(LIB_DIR, "fake_images"), "Fake", "image/jpeg"),
    ("Real Videos", os.path.join(LIB_DIR, "real_videos"), "Real", "video/mp4"),
    ("Fake Videos", os.path.join(LIB_DIR, "fake_videos"), "Fake", "video/mp4"),
]

def main():
    print("=" * 85, flush=True)
    print("      DEEPFAKE DETECTOR BENCHMARK EVALUATION (131 MEDIA FILES)", flush=True)
    print("=" * 85, flush=True)

    overall_total = 0
    overall_correct = 0
    cat_stats = {}
    detailed_results = []

    start_time = time.time()

    for cat_name, folder, expected, default_mime in categories:
        if not os.path.isdir(folder):
            print(f"Skipping {cat_name}: {folder} not found", flush=True)
            continue
        
        files = sorted([f for f in os.listdir(folder) if not f.startswith(".") and f != "README.md"])
        print(f"\n--> Evaluating {cat_name} ({len(files)} files | Expected: {expected})...", flush=True)
        
        cat_correct = 0
        cat_total = 0

        for fname in files:
            fpath = os.path.join(folder, fname)
            if not os.path.isfile(fpath):
                continue

            mime = default_mime
            if fname.lower().endswith(('.png',)):
                mime = 'image/png'
            elif fname.lower().endswith(('.jpg', '.jpeg')):
                mime = 'image/jpeg'
            elif fname.lower().endswith(('.mp4',)):
                mime = 'video/mp4'
            elif fname.lower().endswith(('.mov',)):
                mime = 'video/quicktime'

            cat_total += 1
            overall_total += 1
            t0 = time.time()

            try:
                with open(fpath, "rb") as f:
                    resp = requests.post(
                        API_URL,
                        files={"file": (fname, f, mime)},
                        timeout=120
                    )
                elapsed = time.time() - t0

                if resp.status_code == 200:
                    data = resp.json()
                    pred = data.get("prediction", "Unknown")
                    score = data.get("score", 0.0)
                    conf = data.get("confidence", "N/A")
                    is_correct = (pred == expected)
                    if is_correct:
                        cat_correct += 1
                        overall_correct += 1
                    
                    mark = "[PASS]" if is_correct else "[FAIL]"
                    print(f"  {mark} {fname[:40]:<40} | Pred: {pred:<5} | Score: {score:.4f} ({conf}) | {elapsed:.2f}s", flush=True)
                    detailed_results.append({
                        "file": fname,
                        "category": cat_name,
                        "expected": expected,
                        "prediction": pred,
                        "score": score,
                        "confidence": conf,
                        "correct": is_correct,
                        "elapsed": elapsed
                    })
                else:
                    print(f"  [ERR ] {fname[:40]:<40} | HTTP {resp.status_code}", flush=True)
            except Exception as e:
                print(f"  [ERR ] {fname[:40]:<40} | Error: {e}", flush=True)

        cat_acc = (cat_correct / cat_total * 100) if cat_total > 0 else 0
        cat_stats[cat_name] = {"correct": cat_correct, "total": cat_total, "acc": cat_acc}
        print(f"    Subtotal {cat_name}: {cat_correct}/{cat_total} ({cat_acc:.1f}% Accuracy)", flush=True)

    total_elapsed = time.time() - start_time
    total_acc = (overall_correct / overall_total * 100) if overall_total > 0 else 0

    print("\n" + "=" * 85, flush=True)
    print("                           FINAL BENCHMARK SUMMARY", flush=True)
    print("=" * 85, flush=True)
    for cat_name, stats in cat_stats.items():
        print(f"  {cat_name:<16}: {stats['correct']:>2}/{stats['total']:<2} Correct  ({stats['acc']:5.1f}%)", flush=True)
    print("-" * 85, flush=True)
    print(f"  OVERALL TOTAL   : {overall_correct}/{overall_total} Correct  ({total_acc:.1f}% Accuracy) in {total_elapsed:.1f}s", flush=True)
    print("=" * 85, flush=True)

    # Save results to JSON
    out_json = os.path.join(BASE, "benchmark_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"summary": cat_stats, "total_accuracy": total_acc, "results": detailed_results}, f, indent=2)
    print(f"\nDetailed results saved to {out_json}", flush=True)

if __name__ == "__main__":
    main()
