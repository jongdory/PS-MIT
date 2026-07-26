#!/bin/bash
method="psld"
step_size=0.025
device=0

# BraTS - many to one
CUDA_VISIBLE_DEVICES=$device torchrun --standalone --nproc_per_node=1 sample.py \
    --data_path /store4/01.Database/01.Brain/04.BraTS2021/TrainingData \
    --dataset brats --image_size 224 --batch_size 1 \
    --sample_dir ./samples/unet --ode_method euler \
    --resume checkpoint.pth --method $method --missing_channels 1 --source_channels 0 2 --step_size $step_size


# BraTS - many to many
CUDA_VISIBLE_DEVICES=$device torchrun --standalone --nproc_per_node=1 sample.py \
    --data_path /store4/01.Database/01.Brain/04.BraTS2021/TrainingData \
    --dataset brats --image_size 224 --batch_size 1 \
    --sample_dir ./samples/unet --ode_method euler \
    --resume checkpoint.pth --method $method --missing_channels 1 3 --source_channels 0 2 --step_size $step_size