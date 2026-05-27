import numpy as np


def _chunked_mean_op(pred, true, op, chunk_size=1024):
    if pred.shape != true.shape:
        raise ValueError(f"pred and true must have the same shape, got {pred.shape} and {true.shape}")

    total = 0.0
    count = 0
    pred_flat = pred.reshape(pred.shape[0], -1)
    true_flat = true.reshape(true.shape[0], -1)

    for start in range(0, pred_flat.shape[0], chunk_size):
        pred_chunk = pred_flat[start:start + chunk_size]
        true_chunk = true_flat[start:start + chunk_size]
        values = op(pred_chunk, true_chunk)
        total += values.sum(dtype=np.float64)
        count += values.size

    return total / count


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return _chunked_mean_op(pred, true, lambda pred_chunk, true_chunk: np.abs(pred_chunk - true_chunk))


def MSE(pred, true):
    return _chunked_mean_op(pred, true, lambda pred_chunk, true_chunk: (pred_chunk - true_chunk) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    mape = np.abs((pred - true) / true)
    mape = np.where(mape > 5, 0, mape)
    return np.mean(mape)


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe
