import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import TimeSeriesDataset
from model.model_macro_moe import Hi_CSAMacroMoE
from utils.metrics import MAE, MSE


DATASET_PRESETS = {
    "ETTh1": {"data_path": "./ETT/ETTh1.csv", "in_channels": 7, "split_strategy": "ett"},
    "ETTh2": {"data_path": "./ETT/ETTh2.csv", "in_channels": 7, "split_strategy": "ett"},
    "ETTm1": {"data_path": "./ETT/ETTm1.csv", "in_channels": 7, "split_strategy": "ett"},
    "ETTm2": {"data_path": "./ETT/ETTm2.csv", "in_channels": 7, "split_strategy": "ett"},
    "weather": {"data_path": "./weather/weather.csv", "in_channels": 21, "split_strategy": "standard"},
    "electricity": {"data_path": "./electricity/electricity.csv", "in_channels": 321, "split_strategy": "standard"},
}

# Main ablation switch.
# Set True to disable the Macro MoE branch and train/evaluate only the micro backbone.
MICRO_ONLY = True


def get_dataset_preset(data_name):
    if data_name in DATASET_PRESETS:
        return DATASET_PRESETS[data_name]

    data_name_lower = data_name.lower()
    for preset_name, preset in DATASET_PRESETS.items():
        if preset_name.lower() == data_name_lower:
            return preset

    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Hi-CSA with seed-token macro MoE training script")

    parser.add_argument("--model_name", type=str, default="Hi-CSA-MacroMoE", help="Model name used in checkpoint path")
    parser.add_argument("--data_name", type=str, required=True, help="Dataset name")
    parser.add_argument("--data_path", type=str, default=None, help="Dataset path. Defaults from --data_name when known")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints/", help="Checkpoint save directory")
    parser.add_argument("--exp_name", type=str, default="", help="Optional experiment name appended to checkpoint path")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device, e.g. cuda:0 or cpu")

    parser.add_argument("--seq_len", type=int, required=True, help="Input sequence length")
    parser.add_argument("--pred_len", type=int, required=True, help="Prediction length")
    parser.add_argument("--split_strategy", type=str, default=None, choices=["ratio", "standard", "ett"], help="Data split protocol")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="Dataloader workers")

    parser.add_argument("--in_channels", type=int, default=None, help="Input channels. Defaults from --data_name when known")
    parser.add_argument("--d_model", type=int, required=True, help="Hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.0, help="Micro backbone dropout rate")
    parser.add_argument("--kernel_size", type=int, required=True, help="Micro convolution kernel size")
    parser.add_argument("--flourier_k", type=int, required=True, help="Top-k frequencies for micro scale predictor")
    parser.add_argument("--gmm_k", type=int, required=True, help="Top-k sigma values for micro scale predictor")
    parser.add_argument("--num_gaussians", type=int, required=True, help="GMM components")
    parser.add_argument("--num_base", type=int, required=True, help="Basic kernel number")
    parser.add_argument("--max_sigma", type=float, required=True, help="Max sigma")
    parser.add_argument("--use_revin", dest="use_revin", action="store_true", help="Use reversible instance normalization")
    parser.add_argument("--no_revin", dest="use_revin", action="store_false", help="Disable reversible instance normalization")
    parser.set_defaults(use_revin=True)

    parser.add_argument("--macro_num_experts", type=int, default=4, help="Number of shared macro experts")
    parser.add_argument("--macro_top_k", type=int, default=2, help="Number of macro experts activated per sample")
    parser.add_argument("--macro_hidden_dim", type=int, default=None, help="Hidden dimension inside macro experts")
    parser.add_argument("--router_hidden_dim", type=int, default=None, help="Hidden dimension inside the macro router")
    parser.add_argument("--seasonal_top_k", type=int, default=8, help="Top-k FFT frequency components kept by the seasonal expert")
    parser.add_argument("--router_temperature", type=float, default=0.7, help="Softmax temperature for macro expert routing")
    parser.add_argument("--macro_dropout", type=float, default=None, help="Dropout used inside macro experts/router")
    parser.add_argument("--macro_gamma_init", type=float, default=0.01, help="Initial macro residual strength")
    parser.add_argument("--macro_gamma_max", type=float, default=0.1, help="Upper clamp for macro residual gamma")
    parser.add_argument("--macro_condition_max", type=float, default=0.3, help="Maximum FiLM conditioning magnitude for macro experts")
    parser.add_argument("--residual_mode", type=str, default="feature", choices=["none", "output", "feature", "both"], help="Where to apply the initial-input residual")
    parser.add_argument("--micro_only", action="store_true", default=MICRO_ONLY, help="Disable Macro MoE and run only the micro backbone")
    parser.add_argument("--use_macro_moe", dest="micro_only", action="store_false", help="Enable Macro MoE even when MICRO_ONLY is True in run_macro_moe.py")
    parser.add_argument("--lambda_load_balance", type=float, default=0.001, help="Weight for MoE expert load-balancing loss")
    parser.add_argument("--lambda_router_z", type=float, default=0.001, help="Weight for router z-loss stability penalty")
    parser.add_argument("--lambda_router_entropy", type=float, default=0.001, help="Weight for router entropy penalty; higher values make routing sharper")
    parser.add_argument("--lambda_expert_diversity", type=float, default=0.0, help="Weight for expert diversity penalty")
    parser.add_argument("--lambda_macro_aux", type=float, default=0.0, help="Weight for macro-only auxiliary forecast loss")

    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr_factor", type=float, default=0.5, help="LR decay factor when validation loss plateaus")
    parser.add_argument("--lr_patience", type=int, default=2, help="LR scheduler patience in epochs")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience")
    parser.add_argument("--min_delta", type=float, default=1e-4, help="Minimum validation loss improvement")

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

    print(">>> Initializing Model: Hi-CSA seed-token Macro MoE...")
    model = Hi_CSAMacroMoE(
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
        macro_condition_max=args.macro_condition_max,
        residual_mode=args.residual_mode,
        micro_only=args.micro_only,
        use_revin=args.use_revin,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f">>> Trainable Parameters: {total_params / 1e6:.2f} M")
    print(
        f">>> Macro MoE: experts={args.macro_num_experts} | "
        f"top_k={args.macro_top_k} | "
        f"seasonal_top_k={args.seasonal_top_k} | "
        f"router_temperature={args.router_temperature} | "
        f"gamma_init={args.macro_gamma_init} | "
        f"gamma_max={args.macro_gamma_max} | "
        f"condition_max={args.macro_condition_max} | "
        f"residual_mode={args.residual_mode} | "
        f"micro_only={args.micro_only}"
    )
    print(
        f">>> Training objective: MSE + "
        f"{args.lambda_load_balance:g}*LoadBalance + "
        f"{args.lambda_router_z:g}*RouterZ + "
        f"{args.lambda_router_entropy:g}*RouterEntropy + "
        f"{args.lambda_expert_diversity:g}*ExpertSimilarity + "
        f"{args.lambda_macro_aux:g}*MacroAux"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        threshold=args.min_delta,
        threshold_mode="abs",
        min_lr=args.min_lr,
    )

    run_name = f"sl{args.seq_len}_pl{args.pred_len}"
    if args.exp_name:
        run_name = f"{run_name}_{args.exp_name}"
    save_path = os.path.join(args.checkpoints_dir, args.model_name, args.data_name, run_name)
    os.makedirs(save_path, exist_ok=True)
    best_model_path = os.path.join(save_path, "best_model.pth")

    best_val_loss = float("inf")
    patience_counter = 0
    start_time = time.time()

    print(">>> Start Training")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_mse_loss = 0.0
        train_load_balance_loss = 0.0
        train_router_z_loss = 0.0
        train_router_entropy = 0.0
        train_expert_similarity = 0.0
        train_macro_aux_loss = 0.0
        train_macro_update_ratio = 0.0
        train_condition_scale = 0.0
        train_condition_shift = 0.0
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
            load_balance_loss = aux["load_balance_loss"]
            router_z_loss = aux["router_z_loss"]
            router_entropy = aux["router_entropy"]
            expert_similarity = aux["expert_similarity"]
            if args.micro_only:
                macro_aux_loss = mse_loss.new_tensor(0.0)
            else:
                macro_aux_loss = criterion(aux["macro_prediction"], batch_y)
            loss = (
                mse_loss
                + args.lambda_load_balance * load_balance_loss
                + args.lambda_router_z * router_z_loss
                + args.lambda_router_entropy * router_entropy
                + args.lambda_expert_diversity * expert_similarity
                + args.lambda_macro_aux * macro_aux_loss
            )

            if torch.isnan(loss):
                raise RuntimeError(f"NaN loss detected at epoch {epoch}")

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_mse_loss += mse_loss.item()
            train_load_balance_loss += load_balance_loss.item()
            train_router_z_loss += router_z_loss.item()
            train_router_entropy += router_entropy.item()
            train_expert_similarity += expert_similarity.item()
            train_macro_aux_loss += macro_aux_loss.item()
            train_macro_update_ratio += aux["macro_update_ratio"].item()
            train_condition_scale += aux["condition_scale_mean"].item()
            train_condition_shift += aux["condition_shift_mean"].item()
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
                "lb": load_balance_loss.item(),
                "rz": router_z_loss.item(),
                "ent": router_entropy.item(),
                "maux": macro_aux_loss.item(),
            })

        avg_train_loss = train_loss / max(1, count)
        avg_train_mse_loss = train_mse_loss / max(1, count)
        avg_load_balance_loss = train_load_balance_loss / max(1, count)
        avg_router_z_loss = train_router_z_loss / max(1, count)
        avg_router_entropy = train_router_entropy / max(1, count)
        avg_expert_similarity = train_expert_similarity / max(1, count)
        avg_macro_aux_loss = train_macro_aux_loss / max(1, count)
        avg_macro_update_ratio = train_macro_update_ratio / max(1, count)
        avg_condition_scale = train_condition_scale / max(1, count)
        avg_condition_shift = train_condition_shift / max(1, count)
        avg_expert_weights = expert_weight_sum / max(1, sample_count)
        avg_router_probs = router_prob_sum / max(1, sample_count)
        avg_top1_usage = top1_usage_sum / max(1, sample_count)
        avg_topk_usage = topk_usage_sum / max(1, sample_count)
        avg_val_loss, val_mse, val_mae = evaluate(model, val_loader, device, criterion)
        _, test_mse, test_mae = evaluate(model, test_loader, device, criterion)
        lr_before = optimizer.param_groups[0]["lr"]
        gamma = model.effective_macro_gamma().detach().item()
        expert_weight_msg = " | ".join(
            f"{name}:{weight:.3f}"
            for name, weight in zip(model.macro_expert_names(), avg_expert_weights.tolist())
        )
        router_prob_msg = " | ".join(
            f"{name}:{prob:.3f}"
            for name, prob in zip(model.macro_expert_names(), avg_router_probs.tolist())
        )
        top1_usage_msg = " | ".join(
            f"{name}:{usage:.3f}"
            for name, usage in zip(model.macro_expert_names(), avg_top1_usage.tolist())
        )
        topk_usage_msg = " | ".join(
            f"{name}:{usage:.3f}"
            for name, usage in zip(model.macro_expert_names(), avg_topk_usage.tolist())
        )

        print(
            f"Epoch [{epoch}/{args.epochs}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Train MSE: {avg_train_mse_loss:.4f} | "
            f"LoadBalance: {avg_load_balance_loss:.6g} | "
            f"RouterZ: {avg_router_z_loss:.6g} | "
            f"RouterEnt: {avg_router_entropy:.6g} | "
            f"ExpertSim: {avg_expert_similarity:.4f} | "
            f"MacroAux: {avg_macro_aux_loss:.4f} | "
            f"MacroRatio: {avg_macro_update_ratio:.4f} | "
            f"CondScale: {avg_condition_scale:.4f} | "
            f"CondShift: {avg_condition_shift:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val MSE: {val_mse:.4f} | "
            f"Val MAE: {val_mae:.4f} | "
            f"Test MSE: {test_mse:.4f} | "
            f"Test MAE: {test_mae:.4f} | "
            f"Gamma: {gamma:.4f} | "
            f"LR: {lr_before:.2e}"
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
