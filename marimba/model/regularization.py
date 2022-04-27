import tensorflow as tf
from typing import Optional

from marimba.model.model_info import Regularizer, Regularizers


def get_regularizers(regularizers: Optional[Regularizers]) -> dict:
    """
    Returns a dictionary of regularizers for the given layer info, to pass as keyword arguments to a layer.

    Parameters
    ----------
    regularizers: Regularizers
        The regularizers to use.

    Returns
    -------
    dict
        A dictionary of regularizers, where the keys are in the set of 'kernel_regularizer', 'bias_regularizer', and 'activity_regularizer', 
        and the values are instances of tf.keras.regularizers.
    """

    if regularizers is None:
        return {}
    else:
        return {
            'kernel_regularizer': get_regularizer(regularizers.kernel),
            'bias_regularizer': get_regularizer(regularizers.bias),
            'activity_regularizer': get_regularizer(regularizers.activity),
        }


def get_regularizer(regularizer: Optional[Regularizer]):
    if regularizer is not None:
        return tf.keras.regularizers.L1L2(l1=regularizer.l1, l2=regularizer.l2)
