export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

model_name='Hi-CSA-PretrainedMacroMoE'
python_bin=${PYTHON_BIN:-python}
data_name=${DATA_NAME:-ETTh1}
data_path=${DATA_PATH:-./ETT/ETTh1.csv}

seq_len=${SEQ_LEN:-96}
pred_len=${PRED_LEN:-192}
batch_size=${BATCH_SIZE:-24}
exp_name=${EXP_NAME:-ett_pretrained_macro_moe_best}
device=${DEVICE:-cuda:0}

macro_num_experts=${MACRO_NUM_EXPERTS:-4}
macro_top_k=${MACRO_TOP_K:-2}
seasonal_top_k=${SEASONAL_TOP_K:-8}
router_temperature=${ROUTER_TEMPERATURE:-1.0}
macro_gamma_init=${MACRO_GAMMA_INIT:-0.01}
macro_gamma_max=${MACRO_GAMMA_MAX:-0.1}
macro_dropout=${MACRO_DROPOUT:-0.24}
macro_proj_init=${MACRO_PROJ_INIT:-random}
macro_target_projection_mode=${MACRO_TARGET_PROJECTION_MODE:-random}
macro_raw_fourier_k=${MACRO_RAW_FOURIER_K:-1}
macro_raw_gamma_init=${MACRO_RAW_GAMMA_INIT:-0.16}
macro_raw_gamma_max=${MACRO_RAW_GAMMA_MAX:-0.8}
lambda_raw_macro_residual=${LAMBDA_RAW_MACRO_RESIDUAL:-0.2}
mae_weight=${MAE_WEIGHT:-0.45}
pretrain_lr=${PRETRAIN_LR:-1e-3}
pretrain_weight_decay=${PRETRAIN_WEIGHT_DECAY:-0.0}
pretrain_cosine_weight=${PRETRAIN_COSINE_WEIGHT:-0.1}
pretrained_finetune_mode=${PRETRAINED_FINETUNE_MODE:-all}
lambda_load_balance=${LAMBDA_LOAD_BALANCE:-0.01}
lambda_router_z=${LAMBDA_ROUTER_Z:-0.02}
lambda_router_entropy=${LAMBDA_ROUTER_ENTROPY:-0.0}
lambda_expert_diversity=${LAMBDA_EXPERT_DIVERSITY:-0.0}
lambda_router_semantic=${LAMBDA_ROUTER_SEMANTIC:-0.0}

default_pretrain_path="./checkpoints/Hi-CSA-PretrainedMacroMoE/ETTh1/sl96_pl96_exp_raw_seasonal_lam005_g015/macro_pretrained_bank.pth"
macro_pretrain_path=${MACRO_PRETRAIN_PATH:-}
pretrain_epochs=${PRETRAIN_EPOCHS:-200}
use_default_pretrain=${USE_DEFAULT_PRETRAIN:-1}
if [ "$use_default_pretrain" = "1" ] && [ -z "$macro_pretrain_path" ] && [ -f "$default_pretrain_path" ]; then
  macro_pretrain_path="$default_pretrain_path"
  pretrain_epochs=${PRETRAIN_EPOCHS:-0}
fi

extra_macro_flags=()
if [ "${MICRO_ONLY:-0}" = "1" ]; then
  extra_macro_flags+=(--micro_only)
else
  extra_macro_flags+=(--use_macro_moe)
fi
if [ "${DISABLE_EXPERT_ADAPTERS:-0}" = "1" ]; then
  extra_macro_flags+=(--disable_expert_adapters)
fi

echo ">>> Script python: $python_bin"
echo ">>> Script seed: ${SEED:-2026}"
echo ">>> Script pred_len: $pred_len"
echo ">>> Script macro_pretrain_path: ${macro_pretrain_path:-<build from train split>}"
echo ">>> Script macro mode: semantic_moe + seasonal_fourier_linear"
echo ">>> Script MoE: experts=$macro_num_experts | top_k=$macro_top_k | seasonal_top_k=$seasonal_top_k | router_temperature=$router_temperature | finetune=$pretrained_finetune_mode"
echo ">>> Script raw macro: fourier_k=$macro_raw_fourier_k | gamma=$macro_raw_gamma_init/$macro_raw_gamma_max | lambda=$lambda_raw_macro_residual | mae_weight=$mae_weight"

$python_bin run_pretrained_macro_moe.py \
  --model_name $model_name \
  --data_name $data_name \
  --data_path $data_path \
  --exp_name $exp_name \
  --seed ${SEED:-2026} \
  \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --split_strategy ett \
  --batch_size $batch_size \
  --num_workers ${NUM_WORKERS:-0} \
  \
  --in_channels 7 \
  --d_model ${D_MODEL:-24} \
  --dropout ${DROPOUT:-0.35} \
  --use_revin \
  --kernel_size ${KERNEL_SIZE:-21} \
  --flourier_k ${FLOURIER_K:-3} \
  --gmm_k ${GMM_K:-4} \
  --num_gaussians ${NUM_GAUSSIANS:-4} \
  --num_base ${NUM_BASE:-8} \
  --max_sigma ${MAX_SIGMA:-70.0} \
  \
  --macro_num_experts $macro_num_experts \
  --macro_top_k $macro_top_k \
  --seasonal_top_k $seasonal_top_k \
  --router_temperature $router_temperature \
  --macro_gamma_init $macro_gamma_init \
  --macro_gamma_max $macro_gamma_max \
  --macro_proj_init $macro_proj_init \
  --macro_fusion_mode residual \
  "${extra_macro_flags[@]}" \
  --macro_raw_residual_mode seasonal_fourier_linear \
  --macro_raw_period 24 \
  --macro_raw_fourier_k $macro_raw_fourier_k \
  --macro_raw_gamma_init $macro_raw_gamma_init \
  --macro_raw_gamma_max $macro_raw_gamma_max \
  --residual_mode feature \
  --macro_dropout $macro_dropout \
  --macro_target_projection_mode $macro_target_projection_mode \
  \
  --pretrain_epochs $pretrain_epochs \
  --pretrain_lr $pretrain_lr \
  --pretrain_weight_decay $pretrain_weight_decay \
  --pretrain_cosine_weight $pretrain_cosine_weight \
  --macro_pretrain_path "$macro_pretrain_path" \
  --pretrained_finetune_mode $pretrained_finetune_mode \
  \
  --lambda_load_balance $lambda_load_balance \
  --lambda_router_z $lambda_router_z \
  --lambda_router_entropy $lambda_router_entropy \
  --lambda_expert_diversity $lambda_expert_diversity \
  --lambda_router_semantic $lambda_router_semantic \
  --lambda_macro_residual 0.0 \
  --lambda_raw_macro_residual $lambda_raw_macro_residual \
  \
  --learning_rate ${LEARNING_RATE:-3e-4} \
  --weight_decay ${WEIGHT_DECAY:-5e-4} \
  --loss_type mse_mae \
  --mae_weight $mae_weight \
  --ema_decay 0.0 \
  --lr_factor 0.3 \
  --lr_patience 4 \
  --min_lr 1e-6 \
  --epochs ${EPOCHS:-100} \
  --patience ${PATIENCE:-8} \
  --min_delta 0 \
  --device $device
