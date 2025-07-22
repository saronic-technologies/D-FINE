# Fine-tuning for 25 epochs with Exponential-Moving-Average (EMA) decay 0.9992
#
#  * ``epochs`` is reduced from the default so that the run finishes after
#    exactly 25 training epochs.
#  * ``train_dataloader.collate_fn.ema_restart_decay`` controls the EMA
#    decay that is used after the warm-up stage.  We set it to **0.9992** as
#    requested.
#
# Command-line overrides are passed through the ``-u/--update`` flag and take
# precedence over the values coming from the YAML config file.

torchrun \
  --master_port=7777 \
  --nproc_per_node=8 \
  train.py \
  -c configs/dfine/custom/dfine_hgnetv2_l_custom.yml \
  --use-amp \
  --seed=789345 \
  -t dfine_l_obj365_e25.pth \
  -u epochs=25 train_dataloader.collate_fn.ema_restart_decay=0.9992
