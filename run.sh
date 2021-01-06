CUDA_VISIBLE_DEVICES=0 python train_hierarchical.py --sub_rate 0.5 &
CUDA_VISIBLE_DEVICES=1 python train_hierarchical.py --sub_rate 0.5 --loss_mode weight &
CUDA_VISIBLE_DEVICES=2 python train_hierarchical.py --sub_rate 0.5 --loss_mode id &
CUDA_VISIBLE_DEVICES=3 python train_hierarchical.py --sub_rate 0.5 --fusion_mode spade &