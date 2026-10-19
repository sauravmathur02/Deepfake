"""
Download a FREE public deepfake/real image dataset from HuggingFace
Dataset: haywoodsloan/ai-images-deepfake-vs-real  (100% public, no login)
Also tries: dima806/deepfake_vs_real_image_detection

Saves to:
  test_samples/fake_images/
  test_samples/real_images/
"""

import os
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download

# Try this fully public dataset (no gating)
DATASETS_TO_TRY = [
    "haywoodsloan/ai-images-deepfake-vs-real",
    "dima806/deepfake_vs_real_image_detection",
    "Wvolf/DeepFake_Image_Detection",
]

BASE_DIR  = Path(__file__).parent
CACHE_DIR = BASE_DIR / "_hf_cache"
FAKE_DIR  = BASE_DIR / "fake_images"
REAL_DIR  = BASE_DIR / "real_images"

os.makedirs(FAKE_DIR, exist_ok=True)
os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def try_download(repo_id):
    print(f"\n  Trying: {repo_id}")
    try:
        local_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            cache_dir=str(CACHE_DIR),
            ignore_patterns=["*.parquet", "*.arrow", "*.safetensors"],
        )
        return local_path
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return None

def organize_images(local_path):
    copied_fake = 0
    copied_real = 0
    print(f"\n  Organizing from: {local_path}")
    for root, dirs, files in os.walk(local_path):
        root_lower = root.lower().replace("\\", "/")
        for fname in files:
            if Path(fname).suffix.lower() not in IMG_EXTS:
                continue
            src = Path(root) / fname

            is_fake = any(k in root_lower for k in ["fake", "deepfake", "synthetic", "generated", "ai"])
            is_real = any(k in root_lower for k in ["real", "authentic", "genuine", "original"])

            if is_fake:
                dst = FAKE_DIR / f"{copied_fake:06d}_{fname}"
                shutil.copy2(src, dst)
                copied_fake += 1
            elif is_real:
                dst = REAL_DIR / f"{copied_real:06d}_{fname}"
                shutil.copy2(src, dst)
                copied_real += 1

            total = copied_fake + copied_real
            if total % 200 == 0 and total > 0:
                print(f"    → {copied_real} real, {copied_fake} fake copied...")

    return copied_real, copied_fake

# --- Main ---
print("=" * 60)
print("  Deepfake Dataset Downloader")
print("=" * 60)

local_path = None
for repo in DATASETS_TO_TRY:
    local_path = try_download(repo)
    if local_path:
        print(f"  ✅ Downloaded: {repo}")
        break

if not local_path:
    print("\n❌ All datasets failed. Showing directory structure debug info.")
    exit(1)

real_count, fake_count = organize_images(local_path)

print("\n" + "=" * 60)
if real_count == 0 and fake_count == 0:
    print("⚠️  Images found but folder names didn't match real/fake.")
    print("   Showing downloaded structure:")
    for root, dirs, files in os.walk(local_path):
        level = root.replace(local_path, '').count(os.sep)
        if level > 4:
            continue
        indent = '  ' * level
        print(f'{indent}{os.path.basename(root)}/')
        if files:
            subindent = '  ' * (level + 1)
            for f in files[:3]:
                print(f'{subindent}{f}')
else:
    print(f"✅ DONE!")
    print(f"   Real images  → test_samples/real_images/  ({real_count} files)")
    print(f"   Fake images  → test_samples/fake_images/  ({fake_count} files)")
    print(f"\n   Now open your project and use Bulk Analysis to test!")
print("=" * 60)
