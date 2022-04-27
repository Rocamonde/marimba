import numpy as np

def flatten_covariates(X: np.ndarray) -> np.ndarray:
    """
    Converts a covariates array indexed as (light_curve, time, passband, channel)
    into a covariates array indexed as (light_curve, time, passband*channel)
    so that it can be processed by the model.
    Effectively, passbands are flattened into channels.
    """
    return X.reshape((X.shape[0], X.shape[1], X.shape[2] * X.shape[3]))


# def zero_pad_light_curves(light_curve_list):
#     """
#     The following function, to be used on a list of light curves, takes a list of light curves and pads them with zeros to make them all the same length.
#     Then returns a numpy array of the padded light curves.
#     Dimensions of the returned array are (number of light curves, number of observations, number of channels)
#     """
#     ts_dim_length = max([d.shape[0] for d in light_curve_list])

#     def get_new_ts(ts):
#         new_ts = np.zeros((ts_dim_length, 3))
#         new_ts[:ts.shape[0], :] = ts
#         return new_ts

#     covariates = np.array([get_new_ts(d) for d in light_curve_list])
#     return covariates
