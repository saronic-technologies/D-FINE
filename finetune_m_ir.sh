#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Fine-tuning helper for the 19-million-parameter D-FINE-M backbone
# ---------------------------------------------------------------------------
#
# This script mirrors the existing `finetune.sh` helper that targets the
# large (L) backbone and adapts it for the medium (M) 19 M-parameter model
# that is provided in `dfine_m_obj365.pth`.
#
# Compared with the original helper we only have to
#   1. point to the correct YAML configuration that is already present in
#      the repository (`configs/dfine/custom/dfine_hgnetv2_m_custom.yml`),
#   2. use the new checkpoint file, and
#   3. slightly reduce the default batch-size related parameters so that
#      the script is a drop-in replacement for 1–8 GPUs.
#
# Usage (single-node multi-GPU):
#   chmod +x finetune_m_ir.sh
#   ./finetune_m_ir.sh            # uses all visible GPUs
#
# The command-line flags are identical to the original helper; pass
# additions/overrides through the `-u/--update` flag.

set -euo pipefail

# Pick an arbitrary—but rarely used—port for torchrun so that concurrent
# experiments do not clash.
MASTER_PORT=${MASTER_PORT:-38617}

# Detect number of visible GPUs so we can run the same script on laptops as
# well as multi-GPU servers.
NUM_GPU=$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)

# Decide whether to launch with GPUs (via `torchrun`) or fall back to a plain
# single-process CPU run when no CUDA device is available.
if [[ "${NUM_GPU}" -eq 0 ]]; then
  echo "[D-FINE] No GPU detected – running on CPU."
  TORCHRUN=(python)
  TORCH_EXTRA_ARGS=(--device cpu)
  NUM_GPU=1  # for batch-size computation below
else
  echo "[D-FINE] Found ${NUM_GPU} GPU(s); launching distributed run …"
  TORCHRUN=(torchrun --master_port=${MASTER_PORT} --nproc_per_node=$NUM_GPU)
  TORCH_EXTRA_ARGS=()
fi

# Launch training.
"${TORCHRUN[@]}" \
  train.py \
  "${TORCH_EXTRA_ARGS[@]}" \
  -c configs/dfine/custom/dfine_hgnetv2_m_custom.yml \
  --use-amp \
  --seed=789345 \
  -t dfine_m_obj365.pth \
  -u \
    epochs=20 \
    train_dataloader.collate_fn.ema_restart_decay=0.9992 \
    train_dataloader.total_batch_size=$((NUM_GPU * 2)) \
    val_dataloader.total_batch_size=$((NUM_GPU * 2)) \
    remap_mscoco_category=False \
    num_classes=15

# Notes
# -----
# • The default training schedule (132 epochs) defined in the YAML is long
#   by design.  For a typical transfer-learning scenario we override it to
#   20 epochs so that the
#   run completes in a reasonable amount of time while still allowing the
#   model to converge on the new data set.
# • Feel free to remove the `-u …` overrides if you want to train for the
#   full duration.
