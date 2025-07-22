#!/usr/bin/env python3
"""eo_ltrb_to_coco.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Utility to convert the proprietary EO/IR object-detection dataset that ships
with the D-FINE examples into standard COCO format so that it can be consumed
directly by the training recipes that expect the paths contained in
`configs/dataset/custom_detection.yml`.

The raw data live in a directory structure similar to::

    /mnt/local/prototype-software-merry/ml-datasets/training/
        └── ir_obj_det/
            └── eo/
                ├── images/
                │   ├── *.jpg | *.png | *.jpeg …
                └── labels/
                    └── *.txt               # One per image.

Each label file contains **one object per line** encoded as::

    <class_name> <left> <top> <right> <bottom>

where the four numbers are floats **normalised to the [0, 1] range** with
respect to the image width/height.

This script copies the images, converts the annotations to COCO JSON, and puts
everything under ``/mnt/local/D-FINE/datasets/ir_obj_det_coco`` so that the
paths inside ``configs/dataset/custom_detection.yml`` become valid.

Example
-------

```bash
python tools/dataset/eo_ltrb_to_coco.py \
    --src /mnt/local/prototype-software-merry/ml-datasets/training/ir_obj_det/eo \
    --dst /mnt/local/D-FINE/datasets/ir_obj_det_coco \
    --split 0.9           # optional train/val split (default 0.8)
```

If you pass ``--no-val``, the entire dataset will be used for training and no
validation annotation will be produced.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def gather_classes(labels_dir: Path) -> list[str]:
    """Walk through *labels_dir* and collect a sorted list of unique classes."""

    classes: set[str] = set()
    for lbl_file in labels_dir.glob("*.txt"):
        for line in lbl_file.read_text().strip().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 1:
                continue
            classes.add(parts[0].lower())

    return sorted(classes)


def parse_ltrb_line(
    line: str,
    img_w: int,
    img_h: int,
    class2id: dict[str, int],
) -> dict | None:
    """Convert a single LTRB-encoded line into a COCO annotation dict."""

    parts = line.strip().split()
    if len(parts) != 5:
        return None

    cls_name, l, t, r, b = parts[0].lower(), *map(float, parts[1:])

    if cls_name not in class2id:
        # Unknown label – silently skip so that the conversion never crashes.
        return None

    # Denormalise to absolute pixel coordinates (COCO expects xywh).
    x = max(0.0, min(l * img_w, img_w - 1))
    y = max(0.0, min(t * img_h, img_h - 1))
    w = max(0.0, min((r - l) * img_w, img_w - x))
    h = max(0.0, min((b - t) * img_h, img_h - y))

    return {
        "bbox": [x, y, w, h],
        "category_id": class2id[cls_name],
        "area": w * h,
        "iscrowd": 0,
        "segmentation": [],
    }


# ---------------------------------------------------------------------------
# Main conversion routine
# ---------------------------------------------------------------------------


def convert_split(
    img_paths: list[Path],
    out_img_dir: Path,
    labels_dir: Path,
    class2id: dict[str, int],
    split_name: str,
):
    """Convert *img_paths* into COCO *images* and *annotations* lists."""

    images = []
    annotations = []
    ann_id = 0

    for img_id, img_path in enumerate(tqdm(img_paths, desc=f"{split_name} images")):
        # Copy the image so that the training code can find it at runtime.
        dest_path = out_img_dir / img_path.name
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, dest_path)

        # Image meta
        with Image.open(img_path) as im:
            width, height = im.size

        images.append(
            {
                "id": img_id,
                "file_name": img_path.name,
                "width": width,
                "height": height,
            }
        )

        # Corresponding txt label.
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue  # allow unlabeled images, just skip

        for line in lbl_path.read_text().splitlines():
            ann = parse_ltrb_line(line, width, height, class2id)
            if ann is None:
                continue
            ann["id"] = ann_id
            ann["image_id"] = img_id
            annotations.append(ann)
            ann_id += 1

    return images, annotations


def write_coco_json(path: Path, images, annotations, categories):
    coco_dataset = {
        "info": {
            "description": "Custom IR/EO object detection dataset",
            "version": "1.0",
            "year": 2025,
        },
        "licenses": [{"id": 1, "name": "unknown", "url": ""}],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(coco_dataset, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--src", required=True, type=Path, help="Path to the raw 'eo' folder (contains images/ and labels/)")
    p.add_argument("--dst", required=True, type=Path, help="Destination directory for the COCO dataset")
    p.add_argument("--split", type=float, default=0.8, help="Train split ratio (default: 0.8)")
    p.add_argument("--no-val", action="store_true", help="Do not create a validation split")
    p.add_argument(
        "--blacklist",
        type=Path,
        default=None,
        help="Optional txt file listing (one per line) image base names (with or without extension) to exclude",
    )

    args = p.parse_args(argv)

    src = args.src.expanduser().resolve()
    dst = args.dst.expanduser().resolve()

    img_dir = src / "images"
    lbl_dir = src / "labels"

    if not img_dir.is_dir() or not lbl_dir.is_dir():
        sys.exit(
            f"Expecting 'images' and 'labels' directories inside {src}. Got {img_dir} and {lbl_dir}."
        )

    # -------------------------------------------------------------------
    # Use the FIXED official class list – order is important to stay in sync
    # with the training configs / model checkpoints.
    # -------------------------------------------------------------------

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

    class2id = {cls_name: idx for idx, cls_name in enumerate(OFFICIAL_CLASSES)}
    categories = [
        {"id": idx, "name": cls_name, "supercategory": cls_name}
        for idx, cls_name in enumerate(OFFICIAL_CLASSES)
    ]

    # Collect images.
    img_paths = sorted(
        [p for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG") for p in img_dir.glob(ext)]
    )

    # -------------------------------------------------------------------
    # Optional blacklist handling – remove user-provided bad samples.
    # Format: plain text, one filename (with or without extension) per line.
    # -------------------------------------------------------------------
    if args.blacklist is not None:
        if not args.blacklist.is_file():
            sys.exit(f"Blacklist file {args.blacklist} not found")

        blacklist_entries = {
            line.strip().lower().split()[0]  # keep only first token if spaces
            for line in args.blacklist.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        }

        def is_blacklisted(path: Path) -> bool:
            base = path.stem.lower()
            return base in blacklist_entries or path.name.lower() in blacklist_entries

        before = len(img_paths)
        img_paths = [p for p in img_paths if not is_blacklisted(p)]
        removed = before - len(img_paths)
        print(f"Blacklist active – removed {removed} images (kept {len(img_paths)})")
    if not img_paths:
        sys.exit(f"No images found in {img_dir}")

    if args.no_val:
        train_paths, val_paths = img_paths, []
    else:
        split_idx = int(len(img_paths) * args.split)
        train_paths, val_paths = img_paths[:split_idx], img_paths[split_idx:]

    # Output folders
    train_img_out = dst / "images" / "train"
    val_img_out = dst / "images" / "val" if val_paths else None

    train_images, train_anns = convert_split(train_paths, train_img_out, lbl_dir, class2id, "train")

    if val_paths:
        val_images, val_anns = convert_split(val_paths, val_img_out, lbl_dir, class2id, "val")
    else:
        val_images, val_anns = [], []

    ann_out_dir = dst / "annotations"
    write_coco_json(ann_out_dir / "instances_train.json", train_images, train_anns, categories)
    if val_paths:
        write_coco_json(ann_out_dir / "instances_val.json", val_images, val_anns, categories)

    print("\n✅ Conversion complete!")
    print(f"   Images copied to       : {dst / 'images'}")
    print(f"   Train annotations path : {ann_out_dir / 'instances_train.json'}")
    if val_paths:
        print(f"   Val annotations path   : {ann_out_dir / 'instances_val.json'}")
    print(f"   Classes ({len(OFFICIAL_CLASSES)})           : {OFFICIAL_CLASSES}")


if __name__ == "__main__":
    main()
