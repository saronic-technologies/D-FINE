torchrun --master_port=7777 --nproc_per_node=8 train.py -c configs/dfine/custom/dfine_hgnetv2_l_custom.yml --use-amp --seed=789345 -t dfine_l_obj365_e25.pth
