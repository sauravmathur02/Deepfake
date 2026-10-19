"""
Extract all images from downloaded train, test, and validation parquet datasets into:
  - test_samples/fake_images/  (label 0)
  - test_samples/real_images/  (label 1)
"""

import os
from pathlib import Path
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).parent
CACHE_DIR = BASE_DIR / "_hf_cache"
FAKE_DIR = BASE_DIR / "fake_images"
REAL_DIR = BASE_DIR / "real_images"

os.makedirs(FAKE_DIR, exist_ok=True)
os.makedirs(REAL_DIR, exist_ok=True)

parquet_files = [
    CACHE_DIR / "train.parquet",
    CACHE_DIR / "test.parquet",
    CACHE_DIR / "val.parquet",
]

total_real = len(list(REAL_DIR.glob("*.png")))
total_fake = len(list(FAKE_DIR.glob("*.png")))

print(f"Existing files: {total_real} real, {total_fake} fake")

# If already extracted train, we can check remaining or re-extract clean
# Let's count and extract any missing or just do test & val
for pf in [CACHE_DIR / "test.parquet", CACHE_DIR / "val.parquet"]:
    if not pf.exists():
        continue
    print(f"Extracting: {pf.name} ...")
    table = pq.read_table(pf)
    df = table.to_pandas()
    
    for _, row in df.iterrows():
        img_data = row['image']
        label = row['label']
        raw_bytes = img_data.get('bytes') if isinstance(img_data, dict) else img_data
        if not raw_bytes:
            continue
        
        if label == 0:
            fn = FAKE_DIR / f"fake_{total_fake:05d}.png"
            total_fake += 1
        else:
            fn = REAL_DIR / f"real_{total_real:05d}.png"
            total_real += 1
            
        with open(fn, 'wb') as f:
            f.write(raw_bytes)

print(f"\n[Dataset Extraction Complete]")
print(f"  Total Real images: {total_real} in {REAL_DIR}")
print(f"  Total Fake images: {total_fake} in {FAKE_DIR}")
print(f"  Combined dataset size: {total_real + total_fake} images")
