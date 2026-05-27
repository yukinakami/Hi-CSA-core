import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import TimeSeriesDataset
from model.model_pretrained_macro_moe import Hi_CSAPretrainedMacroMoE
from model.pretrained_macro_moe import build_global_macro_targets
from utils.metrics import MAE, MSE


DATASET_PRESETS = {
    "ETTh1": {"data_path": "./ETT/ETTh1.csv", "in_channels": 7, "split_strategy": "ett"},
    "ETTh2": {"data_path": "./ETT/ETTh2.csv", "in_channels": 7, "split_strategy": "ett"},
    "ETTm1": {"data_path": "./ETT/ETTm1.csv", "in_channels": 7, "split_strategy": "ett"},
    "ETTm2": {"data_path": "./ETT/ETTm2.csv", "in_channels": 7, "split_strategy": "ett"},
    "weather": {"data_path": "./weather/weather.csv", "in_channels": 21, "split_strategy": "standard"},
    "electricity": {"data_path": "./electricity/electricity.csv", "in_channels": 321, "split_strategy": "standard"},
}

# Main ablation switch for this second macro method. Default run uses macro;
# pass --micro_only for the micro-only ablation.
MICRO_ONLY = False


def get_dataset_preset(data_name):
    if data_name in DATASET_PRESETS:
        return DATASET_PRESETS[data_name]

    data_name_lower = data_name.lower()
    for preset_name, preset in DATASET_PRESETS.items():
        if preset_name.lower() == data_name_lower:
            return preset
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Hi-CSA with train-split pretrained macro MoE experts")

    parser.add_argument("--model_name", type=str, default="Hi-CSA-PretrainedMacroMoE")
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints/")
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--pred_len", type=int, required=True)
    parser.add_argument("--split_strategy", type=str, default=None, choices=["ratio", "standard", "ett"])
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--in_channels", type=int, default=None)
    parser.add_argument("--d_model", type=int, required=True)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--kernel_size", type=int, required=True)
    parser.add_argument("--flourier_k", type=int, required=True)
    parser.add_argument("--gmm_k", type=int, required=True)
    parser.add_argument("--num_gaussians", type=int, required=True)
    parser.add_argument("--num_base", type=int, required=True)
    parser.add_argument("--max_sigma", type=float, required=True)
    parser.add_argument("--use_revin", dest="use_revin", action="store_true")
    parser.add_argument("--no_revin", dest="use_revin", action="store_false")
    parser.set_defaults(use_revin=True)

    parser.add_argument("--macro_num_experts", type=int, default=4)
    parser.add_argument("--macro_top_k", type=int, default=2)
    parser.add_argument("--macro_hidden_dim", type=int, default=None)
    parser.add_argument("--router_hidden_dim", type=int, default=None)
    parser.add_argument("--seasonal_top_k", type=int, default=8)
    parser.add_argument("--router_temperature", type=float, default=1.0)
    parser.add_argument("--macro_dropout", type=float, default=None)
    parser.add_argument("--macro_gamma_init", type=float, default=0.01)
    parser.add_argument("--macro_gamma_max", type=float, default=0.1)
    parser.add_argument(
        "--macro_proj_init",
        type=str,
        default="random",
        choices=["random", "zero", "identity"],
    )
    parser.add_argument(
        "--macro_fusion_mode",
        type=str,
        default="residual",
        choices=["residual", "cross_attention", "residual_cross_attention"],
    )
    parser.add_argument("--macro_cross_heads", type=int, default=12)
    parser.add_argument(
        "--macro_cross_memory",
        type=str,
        default="aggregate",
        choices=["aggregate", "experts", "experts_weighted"],
    )
    parser.add_argument("--macro_cross_gamma_init", type=float, default=0.01)
    parser.add_argument("--macro_cross_gamma_max", type=float, default=0.1)
    parser.add_argument("--use_expert_adapters", action="store_true", default=True)
    parser.add_argument("--disable_expert_adapters", dest="use_expert_adapters", action="store_false")
    parser.add_argument("--macro_output_residual", action="store_true")
    parser.add_argument("--macro_output_gamma_init", type=float, default=0.01)
    parser.add_argument("--macro_output_gamma_max", type=float, default=0.1)
    parser.add_argument(
        "--macro_raw_residual_mode",
        type=str,
        default="none",
        choices=[
            "none",
            "seasonal",
            "trend_seasonal",
            "fourier",
            "seasonal_fourier",
            "trend_seasonal_fourier",
            "seasonal_fourier_linear",
        ],
    )
    parser.add_argument("--macro_raw_period", type=int, default=24)
    parser.add_argument("--macro_raw_fourier_k", type=int, default=4)
    parser.add_argument("--macro_raw_gamma_init", type=float, default=0.01)
    parser.add_argument("--macro_raw_gamma_max", type=float, default=0.1)
    parser.add_argument("--micro_attention_layers", type=int, default=0)
    parser.add_argument("--micro_attention_heads", type=int, default=12)
    parser.add_argument("--micro_attention_gamma_init", type=float, default=0.02)
    parser.add_argument("--micro_attention_gamma_max", type=float, default=0.2)
    parser.add_argument("--micro_attention_causal", action="store_true")
    parser.add_argument("--residual_mode", type=str, default="feature", choices=["none", "output", "feature", "both"])
    parser.add_argument("--micro_only", action="store_true", default=MICRO_ONLY)
    parser.add_argument("--use_macro_moe", dest="micro_only", action="store_false")

    parser.add_argument("--pretrain_epochs", type=int, default=80)
    parser.add_argument("--pretrain_lr", type=float, default=1e-3)
    parser.add_argument("--pretrain_weight_decay", type=float, default=0.0)
    parser.add_argument("--pretrain_cosine_weight", type=float, default=0.1)
    parser.add_argument("--skip_macro_pretrain", action="store_true")
    parser.add_argument("--macro_pretrain_path", type=str, default="")
    parser.add_argument("--freeze_pretrained_experts", action="store_true", default=True)
    parser.add_argument("--finetune_pretrained_experts", dest="freeze_pretrained_experts", action="store_false")
    parser.add_argument(
        "--pretrained_finetune_mode",
        type=str,
        default="seed",
        choices=["none", "seed", "seed_last", "all"],
        help="Fine-tuning scope after expert pretraining when freezing is enabled",
    )
    parser.add_argument("--macro_target_projection_seed", type=int, default=2026)
    parser.add_argument(
        "--macro_target_projection_mode",
        type=str,
        default="random",
        choices=["random", "repeat", "residual"],
        help="How train-split macro targets are mapped into hidden feature space.",
    )

    parser.add_argument("--lambda_load_balance", type=float, default=0.01)
    parser.add_argument("--lambda_router_z", type=float, default=0.0)
    parser.add_argument("--lambda_router_entropy", type=float, default=0.0)
    parser.add_argument("--lambda_expert_diversity", type=float, default=0.0)
    parser.add_argument("--lambda_router_semantic", type=float, default=0.0)
    parser.add_argument("--router_semantic_temperature", type=float, default=1.0)
    parser.add_argument("--lambda_macro_residual", type=float, default=0.0)
    parser.add_argument("--lambda_raw_macro_residual", type=float, default=0.0)

    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--loss_type", type=str, default="mse", choices=["mse", "mse_mae"])
    parser.add_argument("--mae_weight", type=float, default=0.2)
    parser.add_argument("--ema_decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=2)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min_delta", type=float, default=1e-4)

    args = parser.parse_args()
    apply_dataset_preset(args, parser)
    return args


def apply_dataset_preset(args, parser):
    preset = get_dataset_preset(args.data_name)
    if args.data_path is None:
        if preset is None:
            parser.error("--data_path is required when --data_name has no preset")
        args.data_path = preset["data_path"]
    if args.in_channels is None:
        if preset is None:
            parser.error("--in_channels is required when --data_name has no preset")
        args.in_channels = preset["in_channels"]
    if args.split_strategy is None:
        args.split_strategy = preset["split_strategy"] if preset is not None else "ratio"


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unpack_batch(batch):
    if len(batch) == 3:
        batch_x, batch_y, _ = batch
    else:
        batch_x, batch_y = batch
    return batch_x, batch_y


def build_router_semantic_targets(batch_x, temperature=1.0):
    centered = batch_x - batch_x.mean(dim=1, keepdim=True)
    std = centered.std(dim=1, unbiased=False).clamp_min(1e-5)
    normalized = centered / std[:, None, :]

    trend_score = (normalized[:, -1] - normalized[:, 0]).abs().mean(dim=-1)

    spectrum = torch.fft.rfft(normalized, dim=1)
    magnitude = spectrum.abs().mean(dim=-1)
    if magnitude.shape[1] > 1:
        seasonal_energy = magnitude[:, 1:]
        top_k = min(8, seasonal_energy.shape[1])
        top_energy = seasonal_energy.topk(top_k, dim=1).values.sum(dim=1)
        season_score = top_energy / seasonal_energy.sum(dim=1).clamp_min(1e-5)
    else:
        season_score = trend_score.new_zeros(trend_score.shape)

    scale_score = std.mean(dim=-1)
    diff = normalized[:, 1:] - normalized[:, :-1]
    regime_score = diff.std(dim=1, unbiased=False).mean(dim=-1)

    scores = torch.stack([trend_score, season_score, scale_score, regime_score], dim=-1)
    scores = (scores - scores.mean(dim=-1, keepdim=True)) / scores.std(
        dim=-1,
        keepdim=True,
        unbiased=False,
    ).clamp_min(1e-5)
    return F.softmax(scores / max(float(temperature), 1e-5), dim=-1)


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    count = 0
    preds = []
    trues = []

    with torch.no_grad():
        for batch in loader:
            batch_x, batch_y = unpack_batch(batch)
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            if torch.isnan(loss):
                raise RuntimeError("NaN loss detected during evaluation")
            total_loss += loss.item()
            count += 1
            preds.append(outputs.detach().cpu().numpy())
            trues.append(batch_y.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    return total_loss / max(1, count), MSE(preds, trues), MAE(preds, trues)


def init_ema_state(model):
    return {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
        if torch.is_floating_point(tensor)
    }


@torch.no_grad()
def update_ema_state(ema_state, model, decay):
    current_state = model.state_dict()
    for name, ema_tensor in ema_state.items():
        ema_tensor.mul_(decay).add_(current_state[name].detach(), alpha=1.0 - decay)


@torch.no_grad()
def swap_floating_state(model, source_state):
    backup = {}
    current_state = model.state_dict()
    for name, source_tensor in source_state.items():
        target_tensor = current_state[name]
        backup[name] = target_tensor.detach().clone()
        target_tensor.copy_(source_tensor.to(device=target_tensor.device, dtype=target_tensor.dtype))
    return backup


def make_state_dict_with_ema(model, ema_state):
    state = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }
    for name, ema_tensor in ema_state.items():
        state[name] = ema_tensor.detach().clone()
    return state


def pretrain_macro_experts(model, train_dataset, args, device, save_path):
    if args.micro_only:
        print(">>> Skip Macro Expert Pretraining")
        return
    if args.macro_pretrain_path:
        print(f">>> Load pretrained macro bank: {args.macro_pretrain_path}")
        bank = torch.load(args.macro_pretrain_path, map_location=device)
        model.macro_moe.load_state_dict(bank["macro_moe_state"])
        if args.freeze_pretrained_experts:
            model.freeze_pretrained_experts(finetune_mode=args.pretrained_finetune_mode)
            if args.pretrained_finetune_mode == "none":
                print(">>> Frozen pretrained expert bank; router/fusion/main model remain trainable")
            else:
                print(
                    f">>> Small fine-tuning enabled for pretrained expert bank "
                    f"({args.pretrained_finetune_mode}); router/fusion/main model remain trainable"
                )
        else:
            print(">>> Pretrained expert bank will be fine-tuned during main training")
        return
    if args.skip_macro_pretrain:
        print(">>> Skip Macro Expert Pretraining")
        return

    print(">>> Stage 1: Build train-split global macro targets")
    macro_targets = build_global_macro_targets(
        train_dataset.global_train_data,
        seq_len=args.seq_len,
        d_model=args.d_model,
        seasonal_top_k=args.seasonal_top_k,
        projection_mode=args.macro_target_projection_mode,
        projection_layer=model.input_residual_proj if args.macro_target_projection_mode == "residual" else None,
        projection_seed=args.macro_target_projection_seed,
        device=device,
    )
    print(
        f">>> Macro targets: {tuple(macro_targets.shape)} from train split only "
        f"(projection={args.macro_target_projection_mode})"
    )

    print(">>> Stage 2: Pretrain semantic macro experts")
    pretrain_params = list(model.macro_moe.seed_tokens.parameters()) if isinstance(model.macro_moe.seed_tokens, nn.Module) else [model.macro_moe.seed_tokens]
    for expert in model.macro_moe.experts:
        pretrain_params.extend(expert.parameters())
    for adapter in model.macro_moe.expert_adapters:
        pretrain_params.extend(adapter.parameters())

    optimizer = torch.optim.Adam(
        pretrain_params,
        lr=args.pretrain_lr,
        weight_decay=args.pretrain_weight_decay,
    )
    expert_names = model.macro_expert_names()

    for epoch in range(1, args.pretrain_epochs + 1):
        model.macro_moe.train()
        optimizer.zero_grad()
        loss, mse_loss, cosine_loss, per_expert_mse = model.macro_moe.pretrain_loss(
            macro_targets,
            cosine_weight=args.pretrain_cosine_weight,
        )
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch == args.pretrain_epochs or epoch % 10 == 0:
            per_expert_msg = " | ".join(
                f"{name}:{value:.4f}" for name, value in zip(expert_names, per_expert_mse.tolist())
            )
            print(
                f"Pretrain [{epoch}/{args.pretrain_epochs}] | "
                f"Loss:{loss.item():.4f} | MSE:{mse_loss.item():.4f} | "
                f"Cos:{cosine_loss.item():.4f} | {per_expert_msg}"
            )

    os.makedirs(save_path, exist_ok=True)
    torch.save(
        {
            "macro_moe_state": model.macro_moe.state_dict(),
            "macro_targets": macro_targets.detach().cpu(),
            "expert_names": expert_names,
        },
        os.path.join(save_path, "macro_pretrained_bank.pth"),
    )
    print(">>> Saved pretrained macro bank")

    if args.freeze_pretrained_experts:
        model.freeze_pretrained_experts(finetune_mode=args.pretrained_finetune_mode)
        if args.pretrained_finetune_mode == "none":
            print(">>> Frozen pretrained expert bank; router/fusion/main model remain trainable")
        else:
            print(
                f">>> Small fine-tuning enabled for pretrained expert bank "
                f"({args.pretrained_finetune_mode}); router/fusion/main model remain trainable"
            )
    else:
        print(">>> Pretrained expert bank will be fine-tuned during main training")


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f">>> Using device: {device}")
    print(">>> Loading Data...")

    train_dataset = TimeSeriesDataset(
        data_path=args.data_path,
        flag="train",
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        scale=True,
        split_strategy=args.split_strategy,
    )
    val_dataset = TimeSeriesDataset(
        data_path=args.data_path,
        flag="val",
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        scale=True,
        scaler=train_dataset.scaler,
        split_strategy=args.split_strategy,
    )
    test_dataset = TimeSeriesDataset(
        data_path=args.data_path,
        flag="test",
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        scale=True,
        scaler=train_dataset.scaler,
        split_strategy=args.split_strategy,
    )

    print(
        f">>> Split: {args.split_strategy} | "
        f"Train samples: {len(train_dataset)} | "
        f"Val samples: {len(val_dataset)} | "
        f"Test samples: {len(test_dataset)}"
    )

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, **loader_kwargs)

    print(">>> Initializing Model: Hi-CSA pretrained semantic Macro MoE...")
    model = Hi_CSAPretrainedMacroMoE(
        in_channels=args.in_channels,
        final_dim=args.d_model,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        flourier_k=args.flourier_k,
        gmm_k=args.gmm_k,
        num_gaussians=args.num_gaussians,
        num_base=args.num_base,
        max_sigma=args.max_sigma,
        dropout=args.dropout,
        kernel_size=args.kernel_size,
        macro_num_experts=args.macro_num_experts,
        macro_top_k=args.macro_top_k,
        macro_hidden_dim=args.macro_hidden_dim,
        router_hidden_dim=args.router_hidden_dim,
        seasonal_top_k=args.seasonal_top_k,
        router_temperature=args.router_temperature,
        macro_dropout=args.macro_dropout,
        macro_gamma_init=args.macro_gamma_init,
        macro_gamma_max=args.macro_gamma_max,
        macro_proj_init=args.macro_proj_init,
        macro_fusion_mode=args.macro_fusion_mode,
        macro_cross_heads=args.macro_cross_heads,
        macro_cross_memory=args.macro_cross_memory,
        macro_cross_gamma_init=args.macro_cross_gamma_init,
        macro_cross_gamma_max=args.macro_cross_gamma_max,
        use_expert_adapters=args.use_expert_adapters,
        macro_output_residual=args.macro_output_residual,
        macro_output_gamma_init=args.macro_output_gamma_init,
        macro_output_gamma_max=args.macro_output_gamma_max,
        macro_raw_residual_mode=args.macro_raw_residual_mode,
        macro_raw_period=args.macro_raw_period,
        macro_raw_fourier_k=args.macro_raw_fourier_k,
        macro_raw_gamma_init=args.macro_raw_gamma_init,
        macro_raw_gamma_max=args.macro_raw_gamma_max,
        micro_attention_layers=args.micro_attention_layers,
        micro_attention_heads=args.micro_attention_heads,
        micro_attention_gamma_init=args.micro_attention_gamma_init,
        micro_attention_gamma_max=args.micro_attention_gamma_max,
        micro_attention_causal=args.micro_attention_causal,
        residual_mode=args.residual_mode,
        micro_only=args.micro_only,
        use_revin=args.use_revin,
    ).to(device)

    run_name = f"sl{args.seq_len}_pl{args.pred_len}"
    if args.exp_name:
        run_name = f"{run_name}_{args.exp_name}"
    save_path = os.path.join(args.checkpoints_dir, args.model_name, args.data_name, run_name)
    os.makedirs(save_path, exist_ok=True)
    best_model_path = os.path.join(save_path, "best_model.pth")

    print(
        f">>> Macro MoE V2: experts={args.macro_num_experts} | top_k={args.macro_top_k} | "
        f"seasonal_top_k={args.seasonal_top_k} | router_temperature={args.router_temperature} | "
        f"gamma_init={args.macro_gamma_init} | gamma_max={args.macro_gamma_max} | "
        f"proj_init={args.macro_proj_init} | fusion={args.macro_fusion_mode} | "
        f"cross_heads={args.macro_cross_heads} | cross_memory={args.macro_cross_memory} | "
        f"cross_gamma_max={args.macro_cross_gamma_max} | "
        f"output_residual={args.macro_output_residual} | "
        f"raw_residual={args.macro_raw_residual_mode} | raw_gamma_max={args.macro_raw_gamma_max} | "
        f"micro_attn_layers={args.micro_attention_layers} | micro_attn_heads={args.micro_attention_heads} | "
        f"micro_attn_gamma_max={args.micro_attention_gamma_max} | "
        f"adapters={args.use_expert_adapters} | "
        f"residual_mode={args.residual_mode} | micro_only={args.micro_only}"
    )
    pretrain_macro_experts(model, train_dataset, args, device, save_path)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f">>> Trainable Parameters After Pretrain Setup: {total_params / 1e6:.2f} M")
    print(
        f">>> Training objective: MSE + "
        f"loss_type={args.loss_type}(mae_weight={args.mae_weight:g}) + "
        f"{args.lambda_load_balance:g}*LoadBalance + "
        f"{args.lambda_router_z:g}*RouterZ + "
        f"{args.lambda_router_entropy:g}*RouterEntropy + "
        f"{args.lambda_expert_diversity:g}*ExpertSimilarity + "
        f"{args.lambda_router_semantic:g}*RouterSemantic + "
        f"{args.lambda_macro_residual:g}*MacroResidual + "
        f"{args.lambda_raw_macro_residual:g}*RawMacroResidual"
    )

    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        threshold=args.min_delta,
        threshold_mode="abs",
        min_lr=args.min_lr,
    )

    best_val_loss = float("inf")
    patience_counter = 0
    start_time = time.time()
    ema_state = init_ema_state(model) if args.ema_decay > 0 else None
    if ema_state is not None:
        print(f">>> EMA enabled: decay={args.ema_decay:g}")

    print(">>> Stage 3/4: Main Hi-CSA Training")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_mse_loss = 0.0
        train_load_balance_loss = 0.0
        train_router_z_loss = 0.0
        train_router_entropy = 0.0
        train_expert_similarity = 0.0
        train_router_semantic = 0.0
        train_macro_residual = 0.0
        train_raw_macro_residual = 0.0
        train_macro_update_ratio = 0.0
        train_effective_macro_ratio = 0.0
        train_macro_cross_ratio = 0.0
        train_effective_macro_cross_ratio = 0.0
        train_macro_cross_alpha_mean = 0.0
        train_macro_cross_alpha_std = 0.0
        train_macro_output_ratio = 0.0
        train_effective_macro_output_ratio = 0.0
        train_raw_macro_ratio = 0.0
        train_effective_raw_macro_ratio = 0.0
        expert_weight_sum = torch.zeros(args.macro_num_experts)
        router_prob_sum = torch.zeros(args.macro_num_experts)
        top1_usage_sum = torch.zeros(args.macro_num_experts)
        topk_usage_sum = torch.zeros(args.macro_num_experts)
        sample_count = 0
        count = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in train_bar:
            batch_x, batch_y = unpack_batch(batch)
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            optimizer.zero_grad()
            outputs, aux = model(batch_x, return_aux=True)
            mse_loss = criterion(outputs, batch_y)
            if args.loss_type == "mse_mae":
                prediction_loss = mse_loss + args.mae_weight * mae_criterion(outputs, batch_y)
            else:
                prediction_loss = mse_loss
            load_balance_loss = aux["load_balance_loss"]
            router_z_loss = aux["router_z_loss"]
            router_entropy = aux["router_entropy"]
            expert_similarity = aux["expert_similarity"]
            if args.lambda_router_semantic > 0:
                semantic_targets = build_router_semantic_targets(
                    batch_x,
                    temperature=args.router_semantic_temperature,
                )
                router_semantic_loss = F.kl_div(
                    aux["router_probs_batch"].clamp_min(1e-8).log(),
                    semantic_targets,
                    reduction="batchmean",
                )
            else:
                router_semantic_loss = mse_loss.new_tensor(0.0)
            if args.lambda_macro_residual > 0:
                residual_target = batch_y - aux["base_prediction"]
                macro_residual_loss = criterion(aux["macro_correction"], residual_target)
            else:
                macro_residual_loss = mse_loss.new_tensor(0.0)
            if args.lambda_raw_macro_residual > 0:
                raw_residual_target = batch_y - aux["base_prediction"]
                raw_macro_residual_loss = criterion(aux["raw_macro_correction"], raw_residual_target)
            else:
                raw_macro_residual_loss = mse_loss.new_tensor(0.0)
            loss = (
                prediction_loss
                + args.lambda_load_balance * load_balance_loss
                + args.lambda_router_z * router_z_loss
                + args.lambda_router_entropy * router_entropy
                + args.lambda_expert_diversity * expert_similarity
                + args.lambda_router_semantic * router_semantic_loss
                + args.lambda_macro_residual * macro_residual_loss
                + args.lambda_raw_macro_residual * raw_macro_residual_loss
            )

            if torch.isnan(loss):
                raise RuntimeError(f"NaN loss detected at epoch {epoch}")

            loss.backward()
            optimizer.step()
            if ema_state is not None:
                update_ema_state(ema_state, model, args.ema_decay)

            train_loss += loss.item()
            train_mse_loss += mse_loss.item()
            train_load_balance_loss += load_balance_loss.item()
            train_router_z_loss += router_z_loss.item()
            train_router_entropy += router_entropy.item()
            train_expert_similarity += expert_similarity.item()
            train_router_semantic += router_semantic_loss.item()
            train_macro_residual += macro_residual_loss.item()
            train_raw_macro_residual += raw_macro_residual_loss.item()
            train_macro_update_ratio += aux["macro_update_ratio"].item()
            train_effective_macro_ratio += aux["effective_macro_ratio"].item()
            train_macro_cross_ratio += aux["macro_cross_ratio"].item()
            train_effective_macro_cross_ratio += aux["effective_macro_cross_ratio"].item()
            train_macro_cross_alpha_mean += aux["macro_cross_alpha_mean"].item()
            train_macro_cross_alpha_std += aux["macro_cross_alpha_std"].item()
            train_macro_output_ratio += aux["macro_output_ratio"].item()
            train_effective_macro_output_ratio += aux["effective_macro_output_ratio"].item()
            train_raw_macro_ratio += aux["raw_macro_ratio"].item()
            train_effective_raw_macro_ratio += aux["effective_raw_macro_ratio"].item()
            batch_size = batch_x.shape[0]
            expert_weight_sum += aux["expert_weights"].detach().cpu() * batch_size
            router_prob_sum += aux["router_probs"].detach().cpu() * batch_size
            top1_usage_sum += aux["top1_usage"].detach().cpu() * batch_size
            topk_usage_sum += aux["topk_usage"].detach().cpu() * batch_size
            sample_count += batch_size
            count += 1

            train_bar.set_postfix({
                "loss": loss.item(),
                "mse": mse_loss.item(),
                "rz": router_z_loss.item(),
                "macro": aux["effective_macro_ratio"].item(),
            })

        avg_train_loss = train_loss / max(1, count)
        avg_train_mse_loss = train_mse_loss / max(1, count)
        avg_load_balance_loss = train_load_balance_loss / max(1, count)
        avg_router_z_loss = train_router_z_loss / max(1, count)
        avg_router_entropy = train_router_entropy / max(1, count)
        avg_expert_similarity = train_expert_similarity / max(1, count)
        avg_router_semantic = train_router_semantic / max(1, count)
        avg_macro_residual = train_macro_residual / max(1, count)
        avg_raw_macro_residual = train_raw_macro_residual / max(1, count)
        avg_macro_update_ratio = train_macro_update_ratio / max(1, count)
        avg_effective_macro_ratio = train_effective_macro_ratio / max(1, count)
        avg_macro_cross_ratio = train_macro_cross_ratio / max(1, count)
        avg_effective_macro_cross_ratio = train_effective_macro_cross_ratio / max(1, count)
        avg_macro_cross_alpha_mean = train_macro_cross_alpha_mean / max(1, count)
        avg_macro_cross_alpha_std = train_macro_cross_alpha_std / max(1, count)
        avg_macro_output_ratio = train_macro_output_ratio / max(1, count)
        avg_effective_macro_output_ratio = train_effective_macro_output_ratio / max(1, count)
        avg_raw_macro_ratio = train_raw_macro_ratio / max(1, count)
        avg_effective_raw_macro_ratio = train_effective_raw_macro_ratio / max(1, count)
        avg_expert_weights = expert_weight_sum / max(1, sample_count)
        avg_router_probs = router_prob_sum / max(1, sample_count)
        avg_top1_usage = top1_usage_sum / max(1, sample_count)
        avg_topk_usage = topk_usage_sum / max(1, sample_count)

        if ema_state is not None:
            live_state = swap_floating_state(model, ema_state)
            avg_val_loss, val_mse, val_mae = evaluate(model, val_loader, device, criterion)
            _, test_mse, test_mae = evaluate(model, test_loader, device, criterion)
            swap_floating_state(model, live_state)
        else:
            avg_val_loss, val_mse, val_mae = evaluate(model, val_loader, device, criterion)
            _, test_mse, test_mae = evaluate(model, test_loader, device, criterion)
        lr_before = optimizer.param_groups[0]["lr"]
        gamma = model.effective_macro_gamma().detach().item()
        expert_names = model.macro_expert_names()
        expert_weight_msg = " | ".join(
            f"{name}:{weight:.3f}" for name, weight in zip(expert_names, avg_expert_weights.tolist())
        )
        router_prob_msg = " | ".join(
            f"{name}:{prob:.3f}" for name, prob in zip(expert_names, avg_router_probs.tolist())
        )
        top1_usage_msg = " | ".join(
            f"{name}:{usage:.3f}" for name, usage in zip(expert_names, avg_top1_usage.tolist())
        )
        topk_usage_msg = " | ".join(
            f"{name}:{usage:.3f}" for name, usage in zip(expert_names, avg_topk_usage.tolist())
        )

        print(
            f"Epoch [{epoch}/{args.epochs}] | "
            f"Train Loss: {avg_train_loss:.4f} | Train MSE: {avg_train_mse_loss:.4f} | "
            f"LoadBalance: {avg_load_balance_loss:.6g} | RouterZ: {avg_router_z_loss:.6g} | "
            f"RouterEnt: {avg_router_entropy:.6g} | RouterSem: {avg_router_semantic:.4f} | "
            f"MacroRes: {avg_macro_residual:.4f} | RawMacroRes: {avg_raw_macro_residual:.4f} | "
            f"ExpertSim: {avg_expert_similarity:.4f} | "
            f"MacroRatio: {avg_macro_update_ratio:.4f} | EffMacroRatio: {avg_effective_macro_ratio:.4f} | "
            f"CrossRatio: {avg_macro_cross_ratio:.4f} | EffCrossRatio: {avg_effective_macro_cross_ratio:.4f} | "
            f"QAlpha: {avg_macro_cross_alpha_mean:.4f}/{avg_macro_cross_alpha_std:.4f} | "
            f"MacroOut: {avg_macro_output_ratio:.4f} | EffMacroOut: {avg_effective_macro_output_ratio:.4f} | "
            f"RawMacro: {avg_raw_macro_ratio:.4f} | EffRawMacro: {avg_effective_raw_macro_ratio:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Val MSE: {val_mse:.4f} | Val MAE: {val_mae:.4f} | "
            f"Test MSE: {test_mse:.4f} | Test MAE: {test_mae:.4f} | "
            f"Gamma: {gamma:.4f} | LR: {lr_before:.2e}"
        )
        print(f"RouterProb | {router_prob_msg}")
        print(f"Top1Usage | {top1_usage_msg}")
        print(f"TopKUsage | {topk_usage_msg}")
        print(f"ActiveWeight | {expert_weight_msg}")

        scheduler.step(avg_val_loss)
        lr_after = optimizer.param_groups[0]["lr"]
        if lr_after < lr_before:
            print(f">>> LR adjusted: {lr_before:.2e} -> {lr_after:.2e}")

        if avg_val_loss < best_val_loss - args.min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            if ema_state is not None:
                torch.save(make_state_dict_with_ema(model, ema_state), best_model_path)
            else:
                torch.save(model.state_dict(), best_model_path)
            print(f">>> Saved Best Model (val mse: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f">>> Early Stopping Triggered (no improvement for {args.patience} epochs)")
                break

    elapsed = time.time() - start_time
    print(f">>> Training Finished. Time: {elapsed:.2f}s")

    print(">>> Start Testing...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_mse, test_mae = evaluate(model, test_loader, device, criterion)
    print(f"Test Result | Loss:{test_loss:.4f} | MSE:{test_mse:.4f} | MAE:{test_mae:.4f}")


if __name__ == "__main__":
    main()
