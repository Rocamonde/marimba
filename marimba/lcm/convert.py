import numpy as np
import pandas as pd
from numpy.typing import NDArray
from avocado.plasticc import plasticc_bands
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from typing import overload, Any, Optional, List, Tuple, Union

default_channels = [
    'flux_mean',
    'flux_variance'
]


@overload
def tabular_to_tensor(lc_tabular: pd.DataFrame, *,
                      channel_labels: Optional[List[str]] = None, 
                      return_obj_ids: Literal[True]) -> Tuple[NDArray[Any], NDArray[np.float64]]: ...


@overload
def tabular_to_tensor(lc_tabular: pd.DataFrame, *,
                      channel_labels: Optional[List[str]] = None, 
                      return_obj_ids: Literal[False]) -> NDArray[np.float64]: ...


def tabular_to_tensor(lc_tabular: pd.DataFrame, *,
                      channel_labels: Optional[List[str]] = None,
                      return_obj_ids: bool = False) -> Union[Tuple[NDArray[Any], NDArray[np.float64]], NDArray[np.float64]]:
    """
    Converts a uniformly sampled light curve dataset from its tabular representation to the tensor representation.
    The dataset must be indexed in the object_id of each light curve.

    Assumes all light curves have the same length and have been sampled uniformly. 
    Channels are indexed in the order they appear in the `channel_labels` list.
    If a list of channel labels is not specified, the `mbm.lcm.convert.default_channels` list is used as the `channel_labels`.

    Args:
        lc_tabular: A pandas dataframe containing the light curves in its tabular representation.
        channel_labels: A list of channel labels.  
        return_obj_ids: If True, returns a tuple of the object_ids and the tensor.

    Returns:
        A tensor of light curves, as a numpy array indexed as (light_curve, time, passband, channel).
    """
    if channel_labels is None:
        channel_labels = default_channels

    object_ids = lc_tabular.index.unique()
    num_objects = len(object_ids)
    num_bands = len(plasticc_bands)
    num_channels = len(channel_labels)
    # TODO will this ever fail?? How to make this rounding safe
    light_curve_length = lc_tabular.shape[0] // num_objects
    lc_tensor = np.zeros(
        (num_objects, light_curve_length, num_bands, num_channels))

    for object_index, object_id in enumerate(object_ids):
        object_df = lc_tabular.loc[object_id]
        for band_index, band in enumerate(plasticc_bands):
            for channel_index, channel in enumerate(channel_labels):
                lc_tensor[object_index, :, band_index,
                          channel_index] = object_df[f'{band}_{channel}']

    if return_obj_ids:
        return np.array(object_ids), lc_tensor
    else:
        return lc_tensor


def tensor_to_tabular(lc_tensor: NDArray, object_ids: NDArray, channel_labels: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Converts a tensor of light curves to its tabular representation, given a list of object_ids whose ordering 
    corresponds to the first axis of the light curves tensor.

    Args:
        lc_tensor: A tensor of light curves, as a numpy array indexed as (light_curve, time, passband, channel).
        object_ids: A list of object_ids whose ordering corresponds to the first axis of the light curves tensor.
        channel_labels: A list of channel labels. If not specified, the `mbm.lcm.convert.default_channels` list is used.

    Returns:
        A pandas dataframe containing the light curves in its tabular representation.
    """

    num_objects, light_curve_length, num_bands, num_channels = lc_tensor.shape
    # create an empty pandas dataframe with num_channels * num_bands columns

    if channel_labels is None:
        channel_labels = default_channels

    if len(channel_labels) != num_channels:
        raise ValueError(
            "Dimension mismatch of the specified channel_labels and the actual number of channels present in the given light curve tensor")

    columns = [
        f'{band}_{channel}' for band in plasticc_bands for channel in channel_labels]

    # We add an extra column for the object_id, then we reset the index.
    lc_tabular = pd.DataFrame(np.zeros((num_objects * light_curve_length,
                              num_channels * num_bands + 1)), columns=[*columns, 'object_id'])

    for object_index, object_id in enumerate(object_ids):
        lc_tabular.loc[object_index * light_curve_length:(object_index + 1) * light_curve_length,
                       'object_id'] = object_id
        for band_index, band in enumerate(plasticc_bands):
            for channel_index, channel in enumerate(channel_labels):
                lc_tabular.loc[object_index * light_curve_length:(object_index + 1) * light_curve_length,
                               f'{band}_{channel}'] = lc_tensor[object_index, :, band_index, channel_index]

    return lc_tabular.set_index('object_id')
