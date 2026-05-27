import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import TimeSeriesDataset
from model.model import Hi_CSA
from utils.metrics import MAE, MSE


DATASET_PRESETS = {
    "ETTh1": {"data_path": "./ETT/ETTh1.csv", "in_channels": 7, "split_strategy": "ett"},
    "ETTh2": {"data_path": "./ETT/ETTh2.csv", "in_channels": 7, "split_strategy": "ett"},
    "ETTm1": {"data_path": "./ETT/ETTm1.csv", "in_channels": 7, "split_strategy": "ett"},
    "ETTm2": {"data_path": "./ETT/ETTm2.csv", "in_channels": 7, "split_strategy": "ett"},
    "weather": {"data_path": "./weather/weather.csv", "in_channels": 21, "split_strategy": "standard"},
    "electricity": {"data_path": "./electricity/electricity.csv", "in_channels": 321, "split_strategy": "standard"},
}


def get_dataset_preset(data_name):
    if data_name in DATASET_PRESETS:
        return DATASET_PRESETS[data_name]

    data_name_lower = data_name.lower()
    for preset_name, preset in DATASET_PRESETS.items():
        if preset_name.lower() == data_name_lower:
            return preset

    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Basic Hi-CSA micro-only training script")

    parser.add_argument("--model_name", type=str, default="Hi-CSA", help="Model name used in checkpoint path")
    parser.add_argument("--data_name", type=str, required=True, help="Dataset name")
    parser.add_argument("--data_path", type=str, default=None, help="Dataset path. Defaults from --data_name when known")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints/", help="Checkpoint save directory")
    parser.add_argument("--exp_name", type=str, default="", help="Optional experiment name appended to checkpoint path")
    parser.add_argument("--seed", type=int, default=1024, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device, e.g. cuda:0 or cpu")

    parser.add_argument("--seq_len", type=int, required=True, help="Input sequence length")
    parser.add_argument("--pred_len", type=int, required=True, help="Prediction length")
    parser.add_argument("--split_strategy", type=str, default=None, choices=["ratio", "standard", "ett"], help="Data split protocol")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="Dataloader workers")

    parser.add_argument("--in_channels", type=int, default=None, help="Input channels. Defaults from --data_name when known")
    parser.add_argument("--d_model", type=int, required=True, help="Hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--kernel_size", type=int, required=True, help="Micro convolution kernel size")
    parser.add_argument("--flourier_k", type=int, required=True, help="Top-k frequencies for micro scale predictor")
    parser.add_argument("--gmm_k", type=int, required=True, help="Top-k sigma values for micro scale predictor")
    parser.add_argument("--num_gaussians", type=int, required=True, help="GMM components")
    parser.add_argument("--num_base", type=int, required=True, help="Basic kernel number")
    parser.add_argument("--max_sigma", type=float, required=True, help="Max sigma")
    parser.add_argument("--use_revin", dest="use_revin", action="store_true", help="Use reversible instance normalization")
    parser.add_argument("--no_revin", dest="use_revin", action="store_false", help="Disable reversible instance normalization")
    parser.set_defaults(use_revin=True)

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

    print(">>> Initializing Model: Hi-CSA micro-only...")
    model = Hi_CSA(
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
        use_revin=args.use_revin,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f">>> Trainable Parameters: {total_params / 1e6:.2f} M")
    print(">>> Training objective: average MSE loss")

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
        count = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in train_bar:
            batch_x, batch_y = unpack_batch(batch)
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            if torch.isnan(loss):
                raise RuntimeError(f"NaN loss detected at epoch {epoch}")

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            count += 1
            train_bar.set_postfix({"mse": loss.item()})

        avg_train_loss = train_loss / max(1, count)
        avg_val_loss, val_mse, val_mae = evaluate(model, val_loader, device, criterion)
        _, test_mse, test_mae = evaluate(model, test_loader, device, criterion)
        lr_before = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch}/{args.epochs}] | "
            f"Train MSE: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val MSE: {val_mse:.4f} | "
            f"Val MAE: {val_mae:.4f} | "
            f"Test MSE: {test_mse:.4f} | "
            f"Test MAE: {test_mae:.4f} | "
            f"LR: {lr_before:.2e}"
        )

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
