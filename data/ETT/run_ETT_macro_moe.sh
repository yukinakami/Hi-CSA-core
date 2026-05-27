export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

model_name='Hi-CSA-MacroMoE'
data_name=${DATA_NAME:-ETTh1}
data_path=${DATA_PATH:-./ETT/ETTh1.csv}

seq_len=${SEQ_LEN:-96}
pred_len=${PRED_LEN:-192}
batch_size=${BATCH_SIZE:-24}
exp_name=${EXP_NAME:-ett_macro_moe_basic}
device=${DEVICE:-cuda:0}

python run_macro_moe.py \
  --model_name $model_name \
  --data_name $data_name \
  --data_path $data_path \
  --exp_name $exp_name \
  \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --split_strategy ett \
  --batch_size $batch_size \
  --num_workers 10 \
  \
  --in_channels 7 \
  --d_model 24 \
  --dropout 0.3 \
  --use_revin \
  --kernel_size 21 \
  --flourier_k 3 \
  --gmm_k 3 \
  --num_gaussians 3 \
  --num_base 8 \
  --max_sigma 70.0 \
  \
  --macro_num_experts ${MACRO_NUM_EXPERTS:-4} \
  --macro_top_k ${MACRO_TOP_K:-2} \
  --seasonal_top_k ${SEASONAL_TOP_K:-8} \
  --router_temperature ${ROUTER_TEMPERATURE:-1.0} \
  --macro_gamma_init ${MACRO_GAMMA_INIT:-0.05} \
  --macro_gamma_max ${MACRO_GAMMA_MAX:-0.2} \
  --macro_condition_max ${MACRO_CONDITION_MAX:-0.3} \
  --residual_mode ${RESIDUAL_MODE:-feature} \
  --macro_dropout ${MACRO_DROPOUT:-0.24} \
  --lambda_load_balance ${LAMBDA_LOAD_BALANCE:-0.0} \
  --lambda_router_z ${LAMBDA_ROUTER_Z:-0.0} \
  --lambda_router_entropy ${LAMBDA_ROUTER_ENTROPY:-0.0} \
  --lambda_expert_diversity ${LAMBDA_EXPERT_DIVERSITY:-0.01} \
  --lambda_macro_aux ${LAMBDA_MACRO_AUX:-0.0} \
  \
  --learning_rate 3e-4 \
  --weight_decay 5e-4 \
  --lr_factor 0.3 \
  --lr_patience 4 \
  --min_lr 1e-6 \
  --epochs ${EPOCHS:-100} \
  --patience 8 \
  --min_delta 0 \
  --device $device
