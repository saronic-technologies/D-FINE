torchrun --master_port=7777 --nproc_per_node=2 train.py -c configs/dfine/custom/dfine_hgnetv2_m_custom.yml --use-amp --seed=0 -t dfine_m_obj365.pth
