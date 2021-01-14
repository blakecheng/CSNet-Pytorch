# CUDA_VISIBLE_DEVICES=0 python train_variation_hierarchical.py --sub_rate 0.5 --batchSize 32 --loss_mode id --weight 0
CUDA_VISIBLE_DEVICES=0 python train_hierarchical.py --sub_rate 0.8 --group_num 1 --loss_mode id_variance_teacher &
CUDA_VISIBLE_DEVICES=1 python train_hierarchical.py --sub_rate 0.9 --group_num 1 --loss_mode id_variance_teacher &
CUDA_VISIBLE_DEVICES=2 python train_hierarchical.py --sub_rate 1.0 --group_num 1 --loss_mode id_variance_teacher &
CUDA_VISIBLE_DEVICES=3 python train_hierarchical.py --sub_rate 2.0 --group_num 1 --loss_mode id_variance_teacher &

CUDA_VISIBLE_DEVICES=4 python train_hierarchical.py --sub_rate 0.8 --group_num 1 --loss_mode id_variance_teacher_fix &
CUDA_VISIBLE_DEVICES=5 python train_hierarchical.py --sub_rate 0.9 --group_num 1 --loss_mode id_variance_teacher_fix &
CUDA_VISIBLE_DEVICES=6 python train_hierarchical.py --sub_rate 1.0 --group_num 1 --loss_mode id_variance_teacher_fix &
CUDA_VISIBLE_DEVICES=7 python train_hierarchical.py --sub_rate 2.0 --group_num 1 --loss_mode id_variance_teacher_fix &
# CUDA_VISIBLE_DEVICES=1 python train_hierarchical.py --sub_rate 0.5 --fusion_mode add &
# CUDA_VISIBLE_DEVICES=2 python train_hierarchical.py --sub_rate 0.5 --fusion_mode add_res &
# CUDA_VISIBLE_DEVICES=3 python train_hierarchical.py --sub_rate 0.5 --fusion_mode spade --group_num 10 &