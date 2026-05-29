#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0

/d/anaconda/python.exe run_global_fft_macro.py \
  --model_name Hi-CSA-GlobalFFT \
  --data_name ETTh1 \
  --exp_name ett_global_fft_macro_96 \
  --seed 2026 \
  --seq_len 96 \
  --pred_len 96 \
  --batch_size 24 \
  --num_workers 10 \
  --d_model 24 \
  --n_heads 12 \
  --dropout 0.24 \
  --kernel_size 20 \
  --flourier_k 3 \
  --gmm_k 3 \
  --macro_k 4 \
  --num_gaussians 3 \
  --num_base 8 \
  --max_sigma 70.0 \
  --macro_dropout 0.24 \
  --cross_dropout 0.24 \
  --cross_gamma_init 0.0 \
  --cross_gamma_limit 0.2 \
  --learning_rate 3e-4 \
  --weight_decay 5e-4 \
  --aux_weight 0.01 \
  --grad_clip 1.0 \
  --lr_factor 0.3 \
  --lr_patience 4 \
  --min_lr 1e-6 \
  --epochs 100 \
  --patience 8 \
  --min_delta 5e-4 \
  --device cuda:0
