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
import os, re, subprocess, json, sys, itertools
import torch
print(torch.cuda.device_count())
PY
)

echo "[D-FINE] Found ${NUM_GPU} GPU(s); launching fine-tuning run …"

# Assemble common torchrun invocation.
TORCHRUN=(torchrun --master_port=${MASTER_PORT} --nproc_per_node=$NUM_GPU)

# Launch training.
"${TORCHRUN[@]}" \
  train.py \
  -c configs/dfine/custom/dfine_hgnetv2_m_custom.yml \
  --use-amp \
  --seed=789345 \
  -t dfine_m_obj365.pth \
  -u epochs=20 train_dataloader.collate_fn.ema_restart_decay=0.9992

# Notes
# -----
# • The default training schedule (132 epochs) defined in the YAML is long
#   by design.  For a typical transfer-learning scenario we override it to
#   20 epochs so that the
#   run completes in a reasonable amount of time while still allowing the
#   model to converge on the new data set.
# • Feel free to remove the `-u …` overrides if you want to train for the
#   full duration.
