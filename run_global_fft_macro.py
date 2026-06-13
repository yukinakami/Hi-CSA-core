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
from model.model_global_fft_macro import Hi_CSAGlobalFFTMacro
from utils.metrics import MAE, MSE


DATASET_PRESETS = {
    "ETTh1": {"data_path": "./ETT/ETTh1.csv", "in_channels": 7, "split_strategy": "ett"},
    "ETTh2": {"data_path": "./ETT/ETTh2.csv", "in_channels": 7, "split_strategy": "ett"},
    "ETTm1": {"data_path": "./ETT/ETTm1.csv", "in_channels": 7, "split_strategy": "ett"},
    "ETTm2": {"data_path": "./ETT/ETTm2.csv", "in_channels": 7, "split_strategy": "ett"},
    "weather": {"data_path": "./weather/weather.csv", "in_channels": 21, "split_strategy": "standard"},
    "electricity": {"data_path": "./electricity/electricity.csv", "in_channels": 321, "split_strategy": "standard"},
    "traffic": {"data_path": "./traffic/traffic.csv", "in_channels": 862, "split_strategy": "standard"},
    "exchange_rate": {"data_path": "./exchange_rate/exchange_rate.csv", "in_channels": 8, "split_strategy": "standard"},
}

FORECAST_RESIDUAL_WEIGHT = 0.2


def parse_args():
    parser = argparse.ArgumentParser(description="Hi-CSA with train-split global FFT macro branch")
    parser.add_argument("--model_name", type=str, default="Hi-CSA-GlobalFFT")
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
    parser.add_argument("--n_heads", type=int, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--kernel_size", type=int, required=True)
    parser.add_argument("--flourier_k", type=int, required=True)
    parser.add_argument("--gmm_k", type=int, required=True)
    parser.add_argument("--macro_k", type=int, required=True)
    parser.add_argument("--num_gaussians", type=int, required=True)
    parser.add_argument("--num_base", type=int, required=True)
    parser.add_argument("--max_sigma", type=float, required=True)
    parser.add_argument("--macro_dropout", type=float, default=None)
    parser.add_argument("--cross_dropout", type=float, default=None)
    parser.add_argument("--cross_gamma_init", type=float, default=0.0)
    parser.add_argument("--cross_gamma_limit", type=float, default=0.0)
    parser.add_argument("--use_revin", dest="use_revin", action="store_true")
    parser.add_argument("--no_revin", dest="use_revin", action="store_false")
    parser.set_defaults(use_revin=True)

    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--aux_weight", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr_factor", type=float, default=0.3)
    parser.add_argument("--lr_patience", type=int, default=4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min_delta", type=float, default=5e-4)
    args = parser.parse_args()
    apply_dataset_preset(args, parser)
    return args


def apply_dataset_preset(args, parser):
    preset = DATASET_PRESETS.get(args.data_name)
    if preset is None:
        for name, value in DATASET_PRESETS.items():
            if name.lower() == args.data_name.lower():
                preset = value
                break
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
            outputs, _ = model(batch_x)
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
        f">>> Split: {args.split_strategy} | Train samples: {len(train_dataset)} | "
        f"Val samples: {len(val_dataset)} | Test samples: {len(test_dataset)}"
    )

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, **loader_kwargs)

    print(">>> Initializing Model: Hi-CSA global FFT macro...")
    model = Hi_CSAGlobalFFTMacro(
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
        macro_k=args.macro_k,
        num_heads=args.n_heads,
        macro_dropout=args.macro_dropout,
        cross_dropout=args.cross_dropout,
        cross_gamma_init=args.cross_gamma_init,
        cross_gamma_limit=args.cross_gamma_limit,
        use_revin=args.use_revin,
    ).to(device)
    model.set_global_macro(torch.from_numpy(train_dataset.global_train_data))

    macro_wave_length = model.global_macro_waves.shape[2]
    print(
        f">>> Macro: global FFT | macro_k={args.macro_k} | "
        f"train length={train_dataset.global_train_data.shape[0]} | wave timeline={macro_wave_length} | "
        f"cross_gamma={model.effective_cross_gamma().detach().item():.4f} | "
        f"forecast_residual_gamma={model.effective_forecast_residual_gamma().detach().item():.4f}"
    )
    print(f">>> Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f} M")
    print(f">>> Training objective: MSE + {args.aux_weight:g}*MicroAuxMSE + {FORECAST_RESIDUAL_WEIGHT:g}*ForecastResidualMSE")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
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
        train_main = 0.0
        train_aux = 0.0
        train_forecast_residual = 0.0
        count = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in train_bar:
            batch_x, batch_y = unpack_batch(batch)
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            optimizer.zero_grad()
            outputs, aux = model(batch_x)
            main_loss = criterion(outputs, batch_y)
            aux_loss = criterion(aux["forecast"], batch_y)
            residual_target = batch_y - aux["base_prediction"]
            forecast_residual_loss = criterion(aux["forecast_residual"], residual_target)
            loss = main_loss + args.aux_weight * aux_loss + FORECAST_RESIDUAL_WEIGHT * forecast_residual_loss
            if torch.isnan(loss):
                raise RuntimeError(f"NaN loss detected at epoch {epoch}")
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            train_loss += loss.item()
            train_main += main_loss.item()
            train_aux += aux_loss.item()
            train_forecast_residual += forecast_residual_loss.item()
            count += 1
            train_bar.set_postfix({"loss": loss.item(), "mse": main_loss.item(), "aux": aux_loss.item(), "fres": forecast_residual_loss.item()})

        avg_train_loss = train_loss / max(1, count)
        avg_train_main = train_main / max(1, count)
        avg_train_aux = train_aux / max(1, count)
        avg_train_forecast_residual = train_forecast_residual / max(1, count)
        val_loss, val_mse, val_mae = evaluate(model, val_loader, device, criterion)
        test_loss, test_mse, test_mae = evaluate(model, test_loader, device, criterion)
        lr_before = optimizer.param_groups[0]["lr"]
        gamma = model.effective_cross_gamma().detach().item()
        forecast_residual_gamma = model.effective_forecast_residual_gamma().detach().item()

        print(
            f"Epoch [{epoch}/{args.epochs}] | Train Loss: {avg_train_loss:.4f} | "
            f"Train MSE: {avg_train_main:.4f} | Aux MSE: {avg_train_aux:.4f} | "
            f"ForecastRes MSE: {avg_train_forecast_residual:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val MSE: {val_mse:.4f} | Val MAE: {val_mae:.4f} | "
            f"Test MSE: {test_mse:.4f} | Test MAE: {test_mae:.4f} | "
            f"CrossGamma: {gamma:.4f} | ForecastResGamma: {forecast_residual_gamma:.4f} | LR: {lr_before:.2e}"
        )

        scheduler.step(val_loss)
        lr_after = optimizer.param_groups[0]["lr"]
        if lr_after < lr_before:
            print(f">>> LR adjusted: {lr_before:.2e} -> {lr_after:.2e}")

        if val_loss < best_val_loss - args.min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f">>> Saved Best Model (val loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f">>> Early Stopping Triggered (no improvement for {args.patience} epochs)")
                break

    print(f">>> Training Finished. Time: {time.time() - start_time:.2f}s")
    print(">>> Start Testing...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_mse, test_mae = evaluate(model, test_loader, device, criterion)
    print(f"Test Result | Loss:{test_loss:.4f} | MSE:{test_mse:.4f} | MAE:{test_mae:.4f}")


if __name__ == "__main__":
    main()
