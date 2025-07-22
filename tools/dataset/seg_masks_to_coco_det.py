#!/usr/bin/env python3
"""seg_masks_to_coco_det.py
Convert the EO segmentation dataset (polygon / mask annotations) into a plain
COCO *detection* dataset that is compatible with the D-FINE training recipes
and shares the exact same nine-class mapping as the other converted sets.

Input directory layout (root provided via --src):

    ├── images/        # *.jpg/png
    ├── annotations/   # 1 JSON per image, structure: {width, height, tags, annotations:[{bbox, class_name, ...}, …]}
    └── masks/         # (ignored by this converter)

The converter extracts the *bbox* of every annotation whose *class_name* is one
of the official nine classes, converts it to absolute pixel XYWH, and writes
out COCO-style JSONs plus an images/ copy identical to the detection converter.

Blacklist support: identical semantics to eo_ltrb_to_coco.py – you can pass the
same seg_eo_mislabeled.txt so that both datasets share exclusions.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm


OFFICIAL_CLASSES = [
    "boat",
    "buoy",
    "person",
    "bird",
    "debris",
    "aircraft",
    "helicopter",
    "self",
    "vehicle",
]

CLASS2ID = {name: idx for idx, name in enumerate(OFFICIAL_CLASSES)}


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True, help="Dataset root containing images/ and annotations/")
    p.add_argument("--dst", type=Path, required=True, help="Output directory for COCO dataset")
    p.add_argument("--split", type=float, default=0.8, help="Train split ratio (ignored if --no-val)")
    p.add_argument("--no-val", action="store_true", help="Do not create a validation split")
    p.add_argument("--blacklist", type=Path, default=None, help="Optional blacklist file (one name per line)")
    return p.parse_args(argv)


def load_blacklist(path: Path | None):
    if path is None:
        return set()
    if not path.is_file():
        sys.exit(f"Blacklist file {path} not found")
    entries = {
        line.strip().lower().split()[0]
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    return entries


def convert_images(
    img_paths: list[Path],
    ann_dir: Path,
    out_img_dir: Path,
):
    images = []
    annotations = []
    ann_id = 0

    for img_id, img_path in enumerate(tqdm(img_paths, desc=f"processing {out_img_dir.parent.name}")):
        # Copy image
        dest = out_img_dir / img_path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, dest)

        # Load dimensions
        with Image.open(img_path) as im:
            width, height = im.size

        images.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": width,
            "height": height,
        })

        # Annotation JSON
        ann_path = ann_dir / f"{img_path.stem}.json"
        if not ann_path.is_file():
            continue

        try:
            ann_data = json.loads(ann_path.read_text())
        except Exception as e:
            print(f"⚠️  Could not read annotation {ann_path}: {e}")
            continue

        for ann in ann_data.get("annotations", []):
            cls = ann.get("class_name", "").lower()
            if cls not in CLASS2ID:
                continue

            # bbox is LTRB normalised to 0-1.
            l, t, r, b = ann["bbox"]
            x = max(0.0, min(l * width, width - 1))
            y = max(0.0, min(t * height, height - 1))
            w = max(0.0, min((r - l) * width, width - x))
            h = max(0.0, min((b - t) * height, height - y))

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "bbox": [x, y, w, h],
                "area": w * h,
                "category_id": CLASS2ID[cls],
                "iscrowd": 0,
                "segmentation": [],
            })
            ann_id += 1

    return images, annotations


def write_coco(path: Path, images, annotations):
    coco = {
        "info": {"description": "EO segmentation converted to detection", "version": "1.0", "year": 2025},
        "licenses": [{"id": 1, "name": "unknown", "url": ""}],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": n, "supercategory": n} for i, n in enumerate(OFFICIAL_CLASSES)],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(coco, indent=2))


def main(argv=None):
    args = parse_args(argv)

    img_dir = args.src / "images"
    ann_dir = args.src / "annotations"
    if not img_dir.is_dir() or not ann_dir.is_dir():
        sys.exit(f"Expecting images/ and annotations/ under {args.src}")

    all_imgs = sorted(
        [p for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG") for p in img_dir.glob(ext)]
    )
    if not all_imgs:
        sys.exit(f"No images found in {img_dir}")

    blacklist = load_blacklist(args.blacklist)
    if blacklist:
        before = len(all_imgs)
        all_imgs = [p for p in all_imgs if p.stem.lower() not in blacklist and p.name.lower() not in blacklist]
        print(f"Blacklist removed {before - len(all_imgs)} images, remaining {len(all_imgs)}")

    # Split train/val
    if args.no_val:
        train_imgs, val_imgs = all_imgs, []
    else:
        split_idx = int(len(all_imgs) * args.split)
        train_imgs, val_imgs = all_imgs[:split_idx], all_imgs[split_idx:]

    train_out = args.dst / "images" / "train"
    val_out = args.dst / "images" / "val"

    train_images, train_anns = convert_images(train_imgs, ann_dir, train_out)
    val_images, val_anns = ([], [])
    if val_imgs:
        val_images, val_anns = convert_images(val_imgs, ann_dir, val_out)

    ann_dir_out = args.dst / "annotations"
    write_coco(ann_dir_out / "instances_train.json", train_images, train_anns)
    if val_imgs:
        write_coco(ann_dir_out / "instances_val.json", val_images, val_anns)

    print("✅ Segmentation dataset converted to detection-style COCO")


if __name__ == "__main__":
    main()

