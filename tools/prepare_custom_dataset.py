#!/usr/bin/env python3
"""
Prepare custom dataset for D-FINE training.

This script provides an easy-to-use interface for converting various dataset formats
to COCO format required by D-FINE.
"""

import os
import sys
import argparse
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(
        description='Prepare custom dataset for D-FINE training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert LTRB format dataset
  python prepare_custom_dataset.py \\
    --input-dir /path/to/dataset \\
    --output-dir /path/to/output/coco_dataset

  # Convert entire dataset for training (no validation split)
  python prepare_custom_dataset.py \\
    --input-dir /path/to/dataset/eo \\
    --output-dir /path/to/output/coco_dataset \\
    --no-split

  # Convert with custom train/val split
  python prepare_custom_dataset.py \\
    --input-dir /path/to/dataset/ir \\
    --output-dir /path/to/output/coco_dataset \\
    --train-split 0.9

Notes:
  - Expects dataset structure with images/ and labels/ subdirectories
  - Labels should be in LTRB format: <class_name> <left> <top> <right> <bottom> (normalized)
        """
    )
    
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Path to input dataset directory')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Path to output directory for COCO format dataset')
    parser.add_argument('--format', type=str, default='ltrb', choices=['ltrb', 'yolo'],
                        help='Input dataset format (default: ltrb for normalized left-top-right-bottom)')
    parser.add_argument('--train-split', type=float, default=0.8,
                        help='Ratio of data to use for training (default: 0.8)')
    parser.add_argument('--no-split', action='store_true',
                        help='Use entire dataset for training (no validation split)')
    
    args = parser.parse_args()
    
    # Validate input directory
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        sys.exit(1)
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if args.format in ['ltrb', 'yolo']:
        # Check for expected structure
        images_path = input_path / 'images'
        labels_path = input_path / 'labels'
        
        if not images_path.exists() or not labels_path.exists():
            print(f"Error: Expected 'images' and 'labels' subdirectories in '{args.input_dir}'")
            print("Your dataset should have the following structure:")
            print("  input_dir/")
            print("    ├── images/")
            print("    │   ├── image1.jpg")
            print("    │   └── ...")
            print("    └── labels/")
            print("        ├── image1.txt")
            print("        └── ...")
            sys.exit(1)
        
        # Run the YOLO to COCO converter
        converter_script = Path(__file__).parent / 'convert_yolo_to_coco.py'
        
        cmd = [
            sys.executable,
            str(converter_script),
            '--input-dir', str(args.input_dir),
            '--output-dir', str(args.output_dir),
            '--train-split', str(args.train_split)
        ]
        
        if args.no_split:
            cmd.append('--no-split')
        
        print(f"Converting {args.format.upper()} format dataset to COCO format...")
        print(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error during conversion:")
            print(result.stderr)
            sys.exit(1)
        
        print(result.stdout)
        
        # Print next steps
        print("\n" + "="*60)
        print("Next steps to train D-FINE on your dataset:")
        print("="*60)
        print("\n1. Copy the generated dataset config to D-FINE configs:")
        print(f"   cp {output_path}/dataset_config.yml configs/dataset/custom_detection.yml")
        print("\n2. Train D-FINE on your dataset:")
        print("   # Using COCO pretrained weights (recommended for faster convergence):")
        print("   export model=l  # Choose from: n, s, m, l, x")
        print("   CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 \\")
        print("     train.py -c configs/dfine/dfine_hgnetv2_${model}_coco.yml \\")
        print("     --use-amp --seed=0 -t path/to/dfine_${model}_coco.pth \\")
        print("     --update num_classes={num_classes} \\")
        print("     --update train_dataloader.dataset.img_folder={train_img_folder} \\")
        print("     --update train_dataloader.dataset.ann_file={train_ann_file} \\")
        print("     --update val_dataloader.dataset.img_folder={val_img_folder} \\")
        print("     --update val_dataloader.dataset.ann_file={val_ann_file}")
        print("\n   # Or using Objects365 pretrained weights (best for generalization):")
        print("   CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 \\")
        print("     train.py -c configs/dfine/objects365/dfine_hgnetv2_${model}_obj2coco.yml \\")
        print("     --use-amp --seed=0 -t path/to/dfine_${model}_obj365.pth \\")
        print("     --update ...")
        
        # Get actual paths from the conversion
        train_img_folder = output_path / 'images' / 'train'
        val_img_folder = output_path / 'images' / 'val'
        train_ann_file = output_path / 'annotations' / 'instances_train.json'
        val_ann_file = output_path / 'annotations' / 'instances_val.json'
        
        # Read number of classes from the config
        config_path = output_path / 'dataset_config.yml'
        if config_path.exists():
            with open(config_path, 'r') as f:
                for line in f:
                    if 'num_classes:' in line:
                        num_classes = int(line.split(':')[1].strip())
                        break
        
        print(f"\n   Specific paths for your dataset:")
        print(f"   - num_classes: {num_classes}")
        print(f"   - train_img_folder: {train_img_folder}")
        print(f"   - train_ann_file: {train_ann_file}")
        print(f"   - val_img_folder: {val_img_folder}")
        print(f"   - val_ann_file: {val_ann_file}")
    
    else:
        print(f"Error: Format '{args.format}' is not yet supported")
        sys.exit(1)


if __name__ == '__main__':
    main()