torchrun --master_port=7777 --nproc_per_node=8 train.py -c configs/dfine/custom/dfine_hgnetv2_m_custom.yml --use-amp --seed=7289345 -t dfine_m_obj365.pth
