import os
import re
import json
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

# === Paths ===
BASE_DIR = Path("/home/yazan/VisualLlama/data")
COCO_DIR = BASE_DIR / "coco"

# Minitrain
ZIP_PATH = BASE_DIR / "coco_minitrain_25k.zip"
MINI_DIR = BASE_DIR / "coco_minitrain_25k/images/train2017"  # expected path

# COCO Annotations
ANN_ZIP = COCO_DIR / "annotations_trainval2017.zip"
ANN_DIR = COCO_DIR / "annotations"
FULL_ANN_FILE = ANN_DIR / "annotations" / "instances_train2017.json"  # fixed nesting
OUT_FILE = ANN_DIR / "minitrain2017.json"

# === URLs ===
HF_URL = "https://huggingface.co/datasets/bryanbocao/coco_minitrain/resolve/main/coco_minitrain_25k.zip"
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def download_file(url, dest):
    """Download file with progress bar if it does not already exist."""
    if dest.exists():
        print(f"[✔] {dest.name} already exists — skipping download.")
        return
    print(f"Downloading {url} → {dest}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total_size, unit="B", unit_scale=True, desc="Downloading"
        ) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    print(f"[✔] Downloaded {dest}")


def extract_zip(zip_path, dest_dir):
    """Extract a ZIP file."""
    target_dir = dest_dir / zip_path.stem
    if target_dir.exists() and any(target_dir.rglob("*")):
        print(f"[✔] {target_dir} already extracted, skipping.")
        return
    print(f"Extracting {zip_path} → {dest_dir}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)
    print(f"[✔] Extracted to {dest_dir}")


def get_minitrain_ids(mini_dir):
    """Collect numeric COCO IDs from all .jpg files in train2017."""
    mini_ids = set()
    if not mini_dir.exists():
        print(f"[✘] Expected directory not found: {mini_dir}")
        return mini_ids
    for p in mini_dir.glob("*.jpg"):
        match = re.search(r"(\d+)", p.stem)
        if match:
            mini_ids.add(int(match.group(1)))
    return mini_ids


def filter_annotations():
    """Filter COCO annotations to only include minitrain images."""
    # 1. Get IDs
    mini_ids = get_minitrain_ids(MINI_DIR)
    print(f"[i] Found {len(mini_ids)} minitrain image IDs in {MINI_DIR}")

    if len(mini_ids) < 20000 or len(mini_ids) > 30000:
        print(f"[!] Warning: Expected ~25k images, found {len(mini_ids)}.")

    # 2. Load annotations
    print(f"[i] Loading {FULL_ANN_FILE}")
    with open(FULL_ANN_FILE, "r") as f:
        coco = json.load(f)

    # 3. Filter
    print("[i] Filtering images ...")
    images = [img for img in tqdm(coco["images"], desc="Images") if img["id"] in mini_ids]

    print("[i] Filtering annotations ...")
    anns = [
        ann for ann in tqdm(coco["annotations"], desc="Annotations")
        if ann["image_id"] in mini_ids
    ]

    # 4. Validate
    missing = []
    for img in tqdm(images, desc="Validating"):
        img_path = MINI_DIR / f"{img['id']:012d}.jpg"
        if not img_path.exists():
            missing.append(img_path.name)

    if missing:
        print(f"[!] Warning: {len(missing)} images referenced in annotations are missing.")
        print(f"    Example missing: {missing[:5]}")

    # 5. Save
    coco_mini = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": coco.get("categories", []),
        "images": images,
        "annotations": anns,
    }

    with open(OUT_FILE, "w") as f:
        json.dump(coco_mini, f)

    print(f"[✔] Saved filtered annotations → {OUT_FILE}")
    print(f"    Images: {len(images)} | Annotations: {len(anns)}")


def main():
    COCO_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Minitrain
    download_file(HF_URL, ZIP_PATH)
    if ZIP_PATH.exists():
        extract_zip(ZIP_PATH, BASE_DIR)
    else:
        print("[✘] Failed to download minitrain")

    # Step 2: Annotations
    download_file(COCO_ANN_URL, ANN_ZIP)
    extract_zip(ANN_ZIP, ANN_DIR)

    # Step 3: Filter
    filter_annotations()


if __name__ == "__main__":
    main()
