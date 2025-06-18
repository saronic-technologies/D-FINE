#!/bin/bash
# Quick script to fine-tune D-FINE-M (Medium) model on custom dataset

# Usage: ./finetune_medium_model.sh <input_dataset_dir> <output_dir> [num_gpus]

set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <input_dataset_dir> <output_dir> [num_gpus]"
    echo "Example: $0 ~/data/ir_obj_det/eo ~/models/dfine_custom 4"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"
NUM_GPUS="${3:-4}"  # Default to 4 GPUs

DATASET_NAME=$(basename "$OUTPUT_DIR")
COCO_DIR="${OUTPUT_DIR}/coco_format"
MODEL_DIR="${OUTPUT_DIR}/model"

echo "=== D-FINE-M Fine-tuning Script ==="
echo "Input dataset: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Number of GPUs: $NUM_GPUS"
echo ""

# Step 1: Convert dataset
echo "Step 1: Converting dataset to COCO format..."
python tools/prepare_custom_dataset.py \
    --input-dir "$INPUT_DIR" \
    --output-dir "$COCO_DIR" \
    --no-split

# Step 2: Create directories
echo "Step 2: Setting up directories..."
mkdir -p configs/dfine/custom/objects365
mkdir -p "$MODEL_DIR"

# Step 3: Copy dataset config
echo "Step 3: Copying dataset configuration..."
cp "$COCO_DIR/dataset_config.yml" "configs/dataset/${DATASET_NAME}_detection.yml"

# Step 4: Download pre-trained weights if not exists
if [ ! -f "$MODEL_DIR/dfine_m_obj365.pth" ]; then
    echo "Step 4: Downloading D-FINE-M pre-trained weights..."
    wget -P "$MODEL_DIR" https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_m_obj365.pth
else
    echo "Step 4: Pre-trained weights already exist, skipping download..."
fi

# Step 5: Create custom config
CONFIG_FILE="configs/dfine/custom/objects365/dfine_hgnetv2_m_${DATASET_NAME}.yml"
echo "Step 5: Creating custom configuration..."

cat > "$CONFIG_FILE" << EOF
__include__: [
  '../../../dataset/${DATASET_NAME}_detection.yml',
  '../../../runtime.yml',
  '../../include/dataloader.yml',
  '../../include/optimizer.yml',
  '../../include/dfine_hgnetv2.yml',
]

output_dir: ${MODEL_DIR}/outputs

DFINE:
  backbone: HGNetv2

HGNetv2:
  name: 'B2'
  return_idx: [1, 2, 3]
  freeze_at: -1
  freeze_norm: False
  use_lab: True

DFINETransformer:
  num_layers: 4
  eval_idx: -1

HybridEncoder:
  in_channels: [384, 768, 1536]
  hidden_dim: 256
  depth_mult: 0.67

optimizer:
  type: AdamW
  params:
    -
      params: '^(?=.*backbone)(?!.*norm|bn).*$'
      lr: 0.000025
    -
      params: '^(?=.*backbone)(?=.*norm|bn).*$'
      lr: 0.000025
      weight_decay: 0.
    -
      params: '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn|bias)).*$'
      weight_decay: 0.

  lr: 0.00025
  betas: [0.9, 0.999]
  weight_decay: 0.000125

epochs: 12  # Adjust as needed
train_dataloader:
  dataset:
    transforms:
      policy:
        epoch: 10  # Adjusted for shorter training
  collate_fn:
    stop_epoch: 10
    ema_restart_decay: 0.9999
    base_size_repeat: 6

ema:
  warmups: 0

lr_warmup_scheduler:
  warmup_duration: 0

checkpoint_freq: 1
print_freq: 100
EOF

# Step 6: Start training
echo "Step 6: Starting fine-tuning..."
echo "Configuration file: $CONFIG_FILE"
echo "Pre-trained weights: $MODEL_DIR/dfine_m_obj365.pth"
echo "Output will be saved to: $MODEL_DIR/outputs"
echo ""

if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Running single GPU training..."
    CUDA_VISIBLE_DEVICES=0 python train.py \
        -c "$CONFIG_FILE" \
        --use-amp \
        --seed=0 \
        -t "$MODEL_DIR/dfine_m_obj365.pth"
else
    echo "Running multi-GPU training with $NUM_GPUS GPUs..."
    GPU_LIST=$(seq -s, 0 $((NUM_GPUS-1)))
    CUDA_VISIBLE_DEVICES=$GPU_LIST torchrun \
        --master_port=7777 \
        --nproc_per_node=$NUM_GPUS \
        train.py \
        -c "$CONFIG_FILE" \
        --use-amp \
        --seed=0 \
        -t "$MODEL_DIR/dfine_m_obj365.pth"
fi

echo ""
echo "=== Training Complete ==="
echo "Model checkpoints saved to: $MODEL_DIR/outputs"
echo "Best model: $MODEL_DIR/outputs/model_best.pth"
echo ""
echo "To run inference:"
echo "python tools/inference/torch_inf.py -c $CONFIG_FILE -r $MODEL_DIR/outputs/model_best.pth --input <image_path>"