# Dataset Conversion Guide for D-FINE

This guide explains how to convert your custom dataset to the COCO format required by D-FINE.

## Supported Input Formats

### LTRB Format

The conversion tool supports datasets organized in the following structure:

```
dataset_root/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── labels/
    ├── image1.txt
    ├── image2.txt
    └── ...
```

Where each label file contains annotations in normalized LTRB format:
```
<class_name> <left> <top> <right> <bottom>
```

All coordinates are normalized to [0, 1] range.

Example label file content:
```
self 0.488558 0.913598 0.905581 1.000000
vehicle 0.112048 0.397886 0.137864 0.433302
vehicle 0.155075 0.414886 0.187346 0.455968
person 0.322882 0.489968 0.356587 0.692547
boat 0.434754 0.281722 0.561685 0.487134
```

## Quick Start

### 1. Basic Conversion

```bash
# Navigate to D-FINE directory
cd /home/catid/sources/D-FINE

# Convert your dataset (default is LTRB format)
python tools/prepare_custom_dataset.py \
  --input-dir /path/to/your/dataset/eo \
  --output-dir /path/to/output/coco_dataset
```

### 2. Use Entire Dataset for Training (No Validation Split)

```bash
# Use when you have a separate validation dataset
python tools/prepare_custom_dataset.py \
  --input-dir /path/to/your/dataset/eo \
  --output-dir /path/to/output/coco_dataset \
  --no-split
```

### 3. Custom Train/Validation Split

```bash
# Use 90% for training, 10% for validation
python tools/prepare_custom_dataset.py \
  --input-dir /path/to/your/dataset/ir \
  --output-dir /path/to/output/coco_dataset \
  --train-split 0.9
```

## Output Structure

The conversion creates a COCO-format dataset with the following structure:

```
output_dir/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── val/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
├── annotations/
│   ├── instances_train.json
│   └── instances_val.json
└── dataset_config.yml
```

## Training D-FINE on Your Dataset

After conversion, follow these steps:

### 1. Copy Dataset Config

```bash
cp /path/to/output/coco_dataset/dataset_config.yml configs/dataset/custom_detection.yml
```

### 2. Train from Scratch

```bash
export model=l  # Choose from: n, s, m, l, x

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 \
  train.py -c configs/dfine/custom/dfine_hgnetv2_${model}_custom.yml \
  --use-amp --seed=0
```

### 3. Fine-tune from COCO Pre-trained Weights (Recommended)

```bash
export model=l  # Choose from: n, s, m, l, x

# Download pre-trained weights first
wget https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_${model}_coco.pth

# Fine-tune
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 \
  train.py -c configs/dfine/custom/dfine_hgnetv2_${model}_custom.yml \
  --use-amp --seed=0 -t dfine_${model}_coco.pth
```

### 4. Fine-tune from Objects365 Pre-trained Weights (Best Generalization)

```bash
export model=l  # Choose from: n, s, m, l, x

# Download pre-trained weights first
wget https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_${model}_obj365.pth

# Fine-tune
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 \
  train.py -c configs/dfine/custom/objects365/dfine_hgnetv2_${model}_obj2custom.yml \
  --use-amp --seed=0 -t dfine_${model}_obj365.pth
```

### 5. Fine-tune D-FINE-M (Medium Model) - Detailed Example

For the medium model specifically, here's a complete workflow:

```bash
# 1. Set model size
export model=m

# 2. Convert your dataset (using entire dataset for training)
python tools/prepare_custom_dataset.py \
  --input-dir /path/to/your/dataset \
  --output-dir ~/datasets/custom_coco \
  --no-split

# 3. Download D-FINE-M Objects365 pre-trained weights
wget https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_m_obj365.pth

# 4. Create custom config directory if it doesn't exist
mkdir -p configs/dfine/custom/objects365

# 5. Copy and modify the Objects365-to-COCO config as a template
cp configs/dfine/objects365/dfine_hgnetv2_m_obj2coco.yml \
   configs/dfine/custom/objects365/dfine_hgnetv2_m_obj2custom.yml

# 6. Edit the config to use your dataset
# Open configs/dfine/custom/objects365/dfine_hgnetv2_m_obj2custom.yml and change:
# - Line with "includes:" change "../../../dataset/coco_detection.yml" 
#   to "../../../dataset/custom_detection.yml" (or your dataset config name)

# 7. Copy your dataset config
cp ~/datasets/custom_coco/dataset_config.yml configs/dataset/custom_detection.yml

# 8. Fine-tune the model
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 \
  train.py -c configs/dfine/custom/objects365/dfine_hgnetv2_m_obj2custom.yml \
  --use-amp --seed=0 -t dfine_m_obj365.pth

# For single GPU training:
CUDA_VISIBLE_DEVICES=0 python train.py \
  -c configs/dfine/custom/objects365/dfine_hgnetv2_m_obj2custom.yml \
  --use-amp --seed=0 -t dfine_m_obj365.pth

# 9. Monitor training (the model will save checkpoints periodically)
# Checkpoints will be saved to: outputs/dfine_hgnetv2_m_obj2custom/
```

#### Key Parameters for D-FINE-M:
- **Model Size**: 19M parameters
- **Input Size**: Default 640x640 (configurable)
- **Batch Size**: Default 16 per GPU (adjust based on your GPU memory)
- **Training Epochs**: Default 12 for fine-tuning (can be adjusted in config)
- **Learning Rate**: Automatically scaled based on batch size

#### Adjusting Training Parameters:

If you need to modify training parameters, you can use the `--update` flag:

```bash
# Example: Train for more epochs with smaller batch size
CUDA_VISIBLE_DEVICES=0 python train.py \
  -c configs/dfine/custom/objects365/dfine_hgnetv2_m_obj2custom.yml \
  --use-amp --seed=0 -t dfine_m_obj365.pth \
  --update epochs=25 \
  --update train_dataloader.total_batch_size=8
```

### 6. Testing and Inference with Fine-tuned Model

After fine-tuning, you can test and use your model:

```bash
# Test the fine-tuned model on validation set
CUDA_VISIBLE_DEVICES=0 python train.py \
  -c configs/dfine/custom/objects365/dfine_hgnetv2_m_obj2custom.yml \
  --test-only -r outputs/dfine_hgnetv2_m_obj2custom/model_best.pth

# Run inference on images
python tools/inference/torch_inf.py \
  -c configs/dfine/custom/objects365/dfine_hgnetv2_m_obj2custom.yml \
  -r outputs/dfine_hgnetv2_m_obj2custom/model_best.pth \
  --input /path/to/test/image.jpg \
  --device cuda:0

# Export to ONNX for deployment
python tools/deployment/export_onnx.py \
  -c configs/dfine/custom/objects365/dfine_hgnetv2_m_obj2custom.yml \
  -r outputs/dfine_hgnetv2_m_obj2custom/model_best.pth \
  --check

# Convert to TensorRT for faster inference
trtexec --onnx="model.onnx" --saveEngine="model.engine" --fp16
```

### 7. Tips for Fine-tuning D-FINE-M

1. **GPU Memory**: D-FINE-M requires ~8GB GPU memory with batch size 4. Adjust batch size based on your GPU.

2. **Learning Rate**: The config uses automatic learning rate scaling. If training is unstable, reduce the base learning rate.

3. **Training Time**: With 4 GPUs, expect ~4-6 hours for 12 epochs on a typical dataset.

4. **Early Stopping**: The model saves checkpoints every epoch. Monitor validation metrics to stop early if needed.

5. **Data Augmentation**: The default augmentation policy changes at epoch 48. For shorter training, you might want to adjust this:
   ```bash
   --update train_dataloader.dataset.transforms.policy.epoch=10
   ```

## Advanced Usage

### Custom Class Mapping

If you're using Objects365 pre-trained weights and your dataset only contains specific classes (e.g., 'person', 'vehicle', 'boat'), you can modify the class mapping for faster convergence:

1. Edit `src/solver/_solver.py`
2. Find `self.obj365_ids` and modify it to match your classes:

```python
# Example: If your dataset only has person, cars, and boats
self.obj365_ids = [0, 5, 8]  # Person, Cars, Boat
```

### Single GPU Training

For single GPU training, modify the command:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  -c configs/dfine/custom/dfine_hgnetv2_${model}_custom.yml \
  --use-amp --seed=0
```

### Custom Batch Size

If you need to adjust the batch size due to GPU memory constraints:

1. Create a custom dataloader config or modify the existing one
2. Update `total_batch_size` parameter
3. Adjust learning rates accordingly (linear scaling law)

## Troubleshooting

### Common Issues

1. **"Unknown class" warnings**: This happens when your label files contain class names not found in other files. The script will skip these annotations.

2. **Memory errors during training**: Reduce the batch size or use a smaller model variant (e.g., 's' or 'm' instead of 'l' or 'x').

3. **Poor performance on custom dataset**: Try using pre-trained weights from Objects365 for better generalization.

### Validation

After conversion, verify your dataset:

```bash
# Check the number of images and annotations
python -c "
import json
with open('/path/to/output/coco_dataset/annotations/instances_train.json', 'r') as f:
    data = json.load(f)
    print(f'Classes: {len(data[\"categories\"])}')
    print(f'Train images: {len(data[\"images\"])}')
    print(f'Train annotations: {len(data[\"annotations\"])}')
"
```

## Example: Complete Workflow

```bash
# 1. Convert dataset (using entire dataset for training)
python tools/prepare_custom_dataset.py \
  --input-dir ~/sources/prototype-software-merry/ml-datasets/training/ir_obj_det/eo \
  --output-dir ~/datasets/ir_obj_det_coco \
  --no-split

# 2. Copy config
cp ~/datasets/ir_obj_det_coco/dataset_config.yml configs/dataset/ir_obj_det.yml

# 3. Download pre-trained weights
wget https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_l_obj365.pth

# 4. Create custom training config (based on existing one)
cp configs/dfine/objects365/dfine_hgnetv2_l_obj2coco.yml \
   configs/dfine/custom/dfine_hgnetv2_l_obj2custom.yml

# 5. Edit the custom config to use your dataset config
# Change the dataset include from coco_detection.yml to ir_obj_det.yml

# 6. Train
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 \
  train.py -c configs/dfine/custom/dfine_hgnetv2_l_obj2custom.yml \
  --use-amp --seed=0 -t dfine_l_obj365.pth
```

## Support

For issues or questions:
1. Check the main D-FINE README for general training tips
2. Verify your dataset structure matches the expected format
3. Ensure all dependencies are installed (`pip install -r requirements.txt`)