#!/usr/bin/env python3
"""
Convert custom format dataset to COCO format for D-FINE training.

This script converts a dataset with the following structure:
    dataset_root/
    ├── images/
    └── labels/

Where labels are in normalized LTRB format:
    <class_name> <left> <top> <right> <bottom>

All coordinates are normalized to [0, 1] range.

To COCO format JSON annotations.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description='Convert custom LTRB format dataset to COCO format')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Path to input directory containing images/ and labels/ folders')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Path to output directory for COCO format dataset')
    parser.add_argument('--train-split', type=float, default=0.8,
                        help='Ratio of data to use for training (default: 0.8)')
    parser.add_argument('--no-split', action='store_true',
                        help='Use entire dataset for training (no validation split)')
    return parser.parse_args()


def read_ltrb_label(label_path, img_width, img_height, class_mapping):
    """Read LTRB format label file and convert to COCO format annotations."""
    annotations = []
    
    if not os.path.exists(label_path):
        return annotations
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
            
        class_name = parts[0].lower()  # Normalize to lowercase
        if class_name not in class_mapping:
            print(f"Warning: Unknown class '{class_name}' in {label_path}")
            continue
            
        # LTRB format: left, top, right, bottom (normalized)
        left, top, right, bottom = map(float, parts[1:])
        
        # Convert to COCO format: x, y, width, height (pixels)
        x = left * img_width
        y = top * img_height
        width = (right - left) * img_width
        height = (bottom - top) * img_height
        
        # Ensure coordinates are within image bounds
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        width = max(0, min(width, img_width - x))
        height = max(0, min(height, img_height - y))
        
        annotation = {
            'bbox': [x, y, width, height],
            'category_id': class_mapping[class_name],
            'area': width * height,
            'iscrowd': 0,
            'segmentation': []
        }
        
        annotations.append(annotation)
    
    return annotations


def create_coco_dataset(image_paths, output_images_dir, class_mapping, dataset_name):
    """Create COCO format dataset from image paths."""
    images = []
    annotations = []
    
    image_id = 0
    annotation_id = 0
    
    for img_path in tqdm(image_paths, desc=f'Processing {dataset_name} images'):
        # Ensure img_path is a Path object
        img_path = Path(img_path)
        
        # Copy image to output directory
        img_name = img_path.name
        output_img_path = output_images_dir / img_name
        shutil.copy2(img_path, output_img_path)
        
        # Get image dimensions
        with Image.open(img_path) as img:
            width, height = img.size
        
        # Add image info
        images.append({
            'id': image_id,
            'file_name': img_name,
            'width': width,
            'height': height,
            'license': 1,
            'date_captured': ''
        })
        
        # Get corresponding label file
        label_path = img_path.parent.parent / 'labels' / (img_path.stem + '.txt')
        
        # Read annotations
        anns = read_ltrb_label(label_path, width, height, class_mapping)
        
        for ann in anns:
            ann['id'] = annotation_id
            ann['image_id'] = image_id
            annotations.append(ann)
            annotation_id += 1
        
        image_id += 1
    
    return images, annotations


def main():
    args = parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir = output_dir / 'annotations'
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    if args.no_split:
        # Single training set, no validation
        train_images_dir = output_dir / 'images' / 'train'
        train_images_dir.mkdir(parents=True, exist_ok=True)
        val_images_dir = None
    else:
        # Train/val split
        train_images_dir = output_dir / 'images' / 'train'
        val_images_dir = output_dir / 'images' / 'val'
        train_images_dir.mkdir(parents=True, exist_ok=True)
        val_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all unique classes
    all_classes = set()
    
    labels_dir = input_dir / 'labels'
    if labels_dir.exists():
        for label_file in labels_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        all_classes.add(parts[0].lower())  # Normalize to lowercase
    else:
        print(f"Error: No 'labels' directory found in {input_dir}")
        sys.exit(1)
    
    # Create class mapping
    sorted_classes = sorted(list(all_classes))
    class_mapping = {cls: idx for idx, cls in enumerate(sorted_classes)}
    
    print(f"Found {len(sorted_classes)} unique classes: {sorted_classes}")
    
    # Create categories for COCO format
    categories = []
    for cls_name, cls_id in class_mapping.items():
        categories.append({
            'id': cls_id,
            'name': cls_name,
            'supercategory': cls_name
        })
    
    # Collect all image paths
    all_image_paths = []
    
    images_dir = input_dir / 'images'
    if images_dir.exists():
        for img_ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            all_image_paths.extend(list(images_dir.glob(img_ext)))
    else:
        print(f"Error: No 'images' directory found in {input_dir}")
        sys.exit(1)
    
    # Sort for reproducibility
    all_image_paths.sort()
    
    # Split into train and validation
    num_images = len(all_image_paths)
    
    if args.no_split:
        # Use all data for training
        train_paths = all_image_paths
        val_paths = []
        print(f"Total images: {num_images} (all used for training)")
    else:
        # Split data
        num_train = int(num_images * args.train_split)
        train_paths = all_image_paths[:num_train]
        val_paths = all_image_paths[num_train:]
        print(f"Total images: {num_images}")
        print(f"Train images: {len(train_paths)}")
        print(f"Val images: {len(val_paths)}")
    
    # Create train dataset
    train_images, train_annotations = create_coco_dataset(
        train_paths, train_images_dir, class_mapping, 'train'
    )
    
    # Create val dataset only if we have a split
    if not args.no_split:
        val_images, val_annotations = create_coco_dataset(
            val_paths, val_images_dir, class_mapping, 'val'
        )
    else:
        val_images, val_annotations = [], []
    
    # Create COCO format JSON for train
    train_coco = {
        'info': {
            'description': 'Custom IR/EO Object Detection Dataset',
            'version': '1.0',
            'year': 2025,
            'contributor': 'Custom Dataset',
            'date_created': '2025-01-01'
        },
        'licenses': [
            {
                'id': 1,
                'name': 'Unknown',
                'url': ''
            }
        ],
        'images': train_images,
        'annotations': train_annotations,
        'categories': categories
    }
    
    # Create COCO format JSON for val
    val_coco = {
        'info': {
            'description': 'Custom IR/EO Object Detection Dataset',
            'version': '1.0',
            'year': 2025,
            'contributor': 'Custom Dataset',
            'date_created': '2025-01-01'
        },
        'licenses': [
            {
                'id': 1,
                'name': 'Unknown',
                'url': ''
            }
        ],
        'images': val_images,
        'annotations': val_annotations,
        'categories': categories
    }
    
    # Save JSON files
    train_json_path = annotations_dir / 'instances_train.json'
    
    with open(train_json_path, 'w') as f:
        json.dump(train_coco, f, indent=2)
    
    print(f"\nConversion complete!")
    print(f"Train annotations saved to: {train_json_path}")
    
    if not args.no_split:
        val_json_path = annotations_dir / 'instances_val.json'
        with open(val_json_path, 'w') as f:
            json.dump(val_coco, f, indent=2)
        print(f"Val annotations saved to: {val_json_path}")
    
    print(f"\nDataset statistics:")
    print(f"  Classes: {len(categories)}")
    print(f"  Train: {len(train_images)} images, {len(train_annotations)} annotations")
    if not args.no_split:
        print(f"  Val: {len(val_images)} images, {len(val_annotations)} annotations")
    
    # Create a custom dataset config file
    if args.no_split:
        # Config without validation split
        config_content = f"""task: detection

evaluator:
  type: CocoEvaluator
  iou_types: ['bbox', ]

num_classes: {len(categories)}
remap_mscoco_category: False

train_dataloader:
  type: DataLoader
  dataset:
    type: CocoDetection
    img_folder: {train_images_dir}
    ann_file: {train_json_path}
    return_masks: False
    transforms:
      type: Compose
      ops: ~
  shuffle: True
  num_workers: 4
  drop_last: True
  collate_fn:
    type: BatchImageCollateFunction

# Note: No validation dataloader - use separate validation dataset
"""
    else:
        # Config with validation split
        config_content = f"""task: detection

evaluator:
  type: CocoEvaluator
  iou_types: ['bbox', ]

num_classes: {len(categories)}
remap_mscoco_category: False

train_dataloader:
  type: DataLoader
  dataset:
    type: CocoDetection
    img_folder: {train_images_dir}
    ann_file: {train_json_path}
    return_masks: False
    transforms:
      type: Compose
      ops: ~
  shuffle: True
  num_workers: 4
  drop_last: True
  collate_fn:
    type: BatchImageCollateFunction

val_dataloader:
  type: DataLoader
  dataset:
    type: CocoDetection
    img_folder: {val_images_dir}
    ann_file: {val_json_path}
    return_masks: False
    transforms:
      type: Compose
      ops: ~
  shuffle: False
  num_workers: 4
  drop_last: False
  collate_fn:
    type: BatchImageCollateFunction
"""
    
    config_path = output_dir / 'dataset_config.yml'
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"\nDataset config saved to: {config_path}")
    print("\nYou can now use this dataset with D-FINE by:")
    print(f"1. Copy {config_path} to configs/dataset/")
    print("2. Update your training config to include this dataset config")


if __name__ == '__main__':
    main()