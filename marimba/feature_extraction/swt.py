import pywt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from avocado.plasticc import plasticc_bands

def compute_swt(time_series, band, level=2):
    db1 = pywt.Wavelet('db1')  # type: ignore 
    arr = np.array(time_series[f'{band}_flux_mean'])
    return pywt.swt(arr, db1, level=2)

def _compute_swt(time_series):
    num_bands = len(plasticc_bands)
    swt_coeffs = np.zeros((num_bands, len(time_series) * 4))
    for i, band in enumerate(plasticc_bands):
        (cA2, cD2), (cA1, cD1) = compute_swt(time_series, band)
        swt_coeffs[i, :] = np.concatenate([cA1, cD1, cA2, cD2])
    return swt_coeffs.flatten()


def compute_swt_features(gp_fits):
    covariates = Parallel(n_jobs=10)(delayed(_compute_swt)(time_series) for _, time_series in gp_fits.groupby('object_id'))
    return np.array(covariates)