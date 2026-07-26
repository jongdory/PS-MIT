#!/bin/bash

device=0

CUDA_VISIBLE_DEVICES=$device torchrun --standalone --nproc_per_node=1 train.py \
  --data_path dataset/BraTS2023 \
  --dataset brats --image_size 224 --batch_size 16 --eval_frequency 1 --epochs 10000 \
  --output_dir ./output/unet --lr 1e-4 --decay_lr \
  --resume checkpoint.pth