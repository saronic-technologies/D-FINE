#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Convenience wrapper: convert an Infra-Red (IR) object-detection data set
# that follows the directory layout used inside the prototype-software-merry
# project into COCO format that D-FINE can consume.
# ---------------------------------------------------------------------------
#
# The raw data live under
#   ▸ training/ir_obj_det/ir
#   ▸ evaluation/based-mcap-eval/ir
# (see the user-supplied paths in the project description).
#
# This helper creates a COCO-formatted copy inside
#   /mnt/local/D-FINE/datasets/{ir_obj_det_coco,based_eval_coco}
# and updates `configs/dataset/custom_detection.yml` so that subsequent
# training runs automatically pick up the new IR data.
#
# Example usage:
#   ./tools/prepare_ir_dataset.sh \
#       /mnt/local/prototype-software-merry/ml-datasets/training/ir_obj_det/ir \
#       /mnt/local/prototype-software-merry/ml-datasets/evaluation/based-mcap-eval/ir

set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 <train_ir_root> <eval_ir_root>" >&2
  exit 1
fi

TRAIN_SRC=$1
EVAL_SRC=$2

# Destination directories expected by the existing YAML configuration.
TRAIN_DST=/mnt/local/D-FINE/datasets/ir_obj_det_coco
EVAL_DST=/mnt/local/D-FINE/datasets/based_eval_coco

echo "[IR-DATASET] Preparing training split …"
python tools/prepare_custom_dataset.py \
  --input-dir "$TRAIN_SRC" \
  --output-dir "$TRAIN_DST" \
  --no-split

echo "[IR-DATASET] Preparing evaluation split …"
python tools/prepare_custom_dataset.py \
  --input-dir "$EVAL_SRC" \
  --output-dir "$EVAL_DST" \
  --no-split

echo "[IR-DATASET] All done — COCO-formatted data written to:"
echo "  • $TRAIN_DST"
echo "  • $EVAL_DST"

echo "You can now kick off fine-tuning with ./finetune_m_ir.sh"

