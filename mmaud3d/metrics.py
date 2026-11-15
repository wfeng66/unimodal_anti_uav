import numpy as np


def coordinate_statistics(targets):
    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[1] != 3 or len(targets) == 0:
        raise ValueError("Coordinate targets must be a non-empty Nx3 array.")
    mean = targets.mean(axis=0)
    std = targets.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def localization_metrics(predictions, targets):
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if predictions.shape != targets.shape or predictions.ndim != 2 or predictions.shape[1] != 3:
        raise ValueError("Predictions and targets must have matching Nx3 shapes.")
    if len(targets) == 0:
        raise ValueError("Localization metrics require at least one sample.")
    errors = predictions - targets
    distances = np.linalg.norm(errors, axis=1)
    return {
        "mean_euclidean_error_m": float(np.mean(distances)),
        "median_error_m": float(np.median(distances)),
        "p95_error_m": float(np.percentile(distances, 95)),
        "mae_x_m": float(np.mean(np.abs(errors[:, 0]))),
        "mae_y_m": float(np.mean(np.abs(errors[:, 1]))),
        "mae_z_m": float(np.mean(np.abs(errors[:, 2]))),
        "rmse_3d_m": float(np.sqrt(np.mean(np.sum(errors ** 2, axis=1)))),
        "max_error_m": float(np.max(distances)),
    }
