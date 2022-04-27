import numpy as np


def log_transform(X):
    """
    Covariates array indexed as (light_curve, time, passband, channel) where channel refers to flux or flux error.
    Each channel is normalized separately.
    """
    flux_clip = np.clip(X[:, :, :, 0], 1e-10, 1e10)
    flux_log = np.log(flux_clip)
    return np.concatenate((flux_log[:, :, :, None], X[:, :, :, 1:]), axis=3)


def mean_std(X):
    """
    Covariates array indexed as (light_curve, time, passband, channel) where channel refers to flux or flux error.
    Each channel is normalized separately.
    """
    X_mean = np.mean(X, axis=(1, 2))
    X_std = np.std(X, axis=(1, 2))
    return (X - X_mean[:, None, None, :]) / X_std[:, None, None, :]

    return X


def min_max(X):
    """
    Covariates array indexed as (light_curve, time, passband, channel) where channel refers to flux or flux error.
    Each channel is normalized separately.
    """
    X_min = np.min(X, axis=(1, 2))
    X_max = np.max(X, axis=(1, 2))
    return (X - X_min[:, None, None, :]) / (X_max[:, None, None, :] - X_min[:, None, None, :])
