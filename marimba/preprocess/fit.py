import os

import avocado
import numpy as np
import pandas as pd
from avocado.astronomical_object import AstronomicalObject
from avocado.dataset import Dataset
from avocado.plasticc import plasticc_bands
from avocado.utils import write_dataframe
from marimba.settings import *
from marimba.lcm.convert import tensor_to_tabular


def fit(astronomical_object: AstronomicalObject):
    """
    Parameters
    ----------
    astronomical_object : :class:`AstronomicalObject`
        The astronomical object to featurize.
    return_model : bool
        If true, the light curve model is also returned. Defaults to False.

    Returns
    -------
    raw_features : dict
        The raw extracted features for this object.
    model : dict (optional)
        A dictionary with the light curve model in each band. This is only
        returned if return_model is set to True.
    """
    min_time = astronomical_object.observations['time'].min()
    max_time = astronomical_object.observations['time'].max()

    # Fit the GP and produce an output model
    gp_start_time = min_time - FIT_PAD
    gp_end_time = max_time + FIT_PAD
    gp_times = np.arange(gp_start_time, gp_end_time + 1, FIT_CADENCE)
    gp, _, _ = (
        astronomical_object.fit_gaussian_process()
    )
    gp_fluxes, gp_flux_uncertainties = astronomical_object.predict_gaussian_process(
        plasticc_bands, gp_times, uncertainties=True, fitted_gp=gp
    )

    fit = np.zeros((FIT_TOTAL_LENGTH, len(plasticc_bands), 2))

    sliced_length = min(gp_times.shape[0], FIT_TOTAL_LENGTH)
    fit[:sliced_length, :, 0] = gp_fluxes.T[:sliced_length, :]
    fit[:sliced_length, :, 1] = gp_flux_uncertainties.T[:sliced_length, :]
    return fit


def compute_fits(dataset: Dataset) -> pd.DataFrame:
    """
    Parameters
    ----------
    dataset : :class:`Dataset`
        The dataset to compute the fits for.

    Returns
    -------
    fits : :class:`pd.DataFrame`
        The fits for each object in the dataset.
    """

    fits = []
    object_ids = []
    for object in dataset.objects:  # type: ignore
        fits.append(fit(object))
        object_ids.append(object.metadata['object_id'])
    fits = np.array(fits)
    object_ids = np.array(object_ids)
    return tensor_to_tabular(fits, object_ids=object_ids)


def fit_chunk(dataset_name: str, chunk: int, num_chunks: int, verbose=True, write=True):
    # Load the reference dataset
    if verbose:
        print("Loading reference dataset...")
    dataset = avocado.load(dataset_name, chunk=chunk, num_chunks=num_chunks)

    # Fit the dataset
    if verbose:
        print("Fitting the dataset...")
    fits_df = compute_fits(dataset)

    if verbose:
        print("Writing the fits...")
    fits_filename = f"fits_{dataset_name}.h5"
    fits_path = os.path. join(DATA_FITS_DIR, fits_filename)
    key = FIT_VERSION
    write_dataframe(fits_path, fits_df, key,
                    chunk=chunk, num_chunks=num_chunks)
