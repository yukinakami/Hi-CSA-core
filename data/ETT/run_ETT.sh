export CUDA_VISIBLE_DEVICES=0

model_name='Hi-CSA'
data_name='ETTh1'
data_path='./ETT/ETTh1.csv'

python run.py \
  --model_name $model_name \
  --data_name $data_name \
  --data_path $data_path \
  --exp_name ett_micro_basic \
  \
  --seq_len 96 \
  --pred_len 192 \
  --split_strategy ett \
  --batch_size 24 \
  --num_workers 10 \
  \
  --in_channels 7 \
  --d_model 24 \
  --dropout 0.3 \
  --use_revin \
  --kernel_size 20 \
  --flourier_k 3 \
  --gmm_k 4 \
  --num_gaussians 4 \
  --num_base 8 \
  --max_sigma 70.0 \
  --learning_rate 3e-4 \
  --weight_decay 5e-4 \
  --lr_factor 0.3 \
  --lr_patience 4 \
  --min_lr 1e-6 \
  --epochs 100 \
  --patience 8 \
  --min_delta 0 \
  --device 'cuda:0'
