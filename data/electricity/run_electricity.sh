#!/bin/bash

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

model_name='Hi-CSA'
data_name='electricity'
data_path='./electricity/electricity.csv'

seq_len=${SEQ_LEN:-96}
pred_len=${PRED_LEN:-96}
batch_size=${BATCH_SIZE:-8}
exp_name=${EXP_NAME:-electricity_standard_pl${pred_len}}
device=${DEVICE:-cuda:0}

python run.py \
  --model_name $model_name \
  --data_name $data_name \
  --data_path $data_path \
  --exp_name $exp_name \
  \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --split_strategy standard \
  --batch_size $batch_size \
  --num_workers 2 \
  \
  --in_channels 321 \
  --d_model 24 \
  --dropout 0.2 \
  --use_revin \
  \
  --kernel_size 21 \
  --flourier_k 3 \
  --gmm_k 3 \
  --num_gaussians 3 \
  --num_base 8 \
  --max_sigma 70.0 \
  --learning_rate 3e-4 \
  --weight_decay 5e-4 \
  --lr_factor 0.3 \
  --lr_patience 4 \
  --min_lr 1e-6 \
  --epochs 100 \
  --patience 8 \
  --min_delta 5e-4 \
  --device $device
