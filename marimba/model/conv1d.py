from marimba.model.utils import conv_block, dense_block, global_pool_block, residual_block
import tensorflow as tf
from marimba.lcm.tensor import flatten_covariates
from marimba.model.losses import *
from marimba.model.metrics import *
from marimba.model.model_info import ModelInfo
from marimba.preprocess.normalize import mean_std


def build_block_stack(x, layers, block_fn, dim=1):
    """
    Builds a block stack.
    """
    if isinstance(layers, list):
        for layer in layers:
            x = block_fn(x, layer, dim=dim)
    else:
        number = layers.number
        if number is None:
            raise ValueError(
                'number of layers must be specified, or provide a list of layers.')
        for _ in range(number):
            x = block_fn(x, layers, dim=dim)
    return x


def build_model(model_info: ModelInfo, verbose=False):
    """
    Builds a model from a model_info object.

    Parameters
    ----------
    model_info: ModelInfo
        A model_info object.
    verbose: bool
        Whether to print the model summary.

    Returns
    -------
    model: tf.keras.Model
        A compiled model.

    Raises
    ------
    ValueError
        If the model_info object is not valid.
    """
    if model_info.build.model not in ['resnet1d', 'conv1d', 'conv2d', 'resnet2d']:
        raise ValueError(f'Model {model_info.build.model} not supported')
    dim = 1 if model_info.build.model.endswith('1d') else 2

    num_passbands = model_info.build.num_passbands
    time_series_dim = model_info.build.time_series_dim
    layers = model_info.build.layers
    num_classes = model_info.build.num_classes

    if layers.conv is None:
        raise ValueError('Convolutional layers must be specified.')
    
    if dim == 1:
        input = tf.keras.layers.Input(
            shape=(None, num_passbands * time_series_dim))
    else:
        input = tf.keras.layers.Input(
            shape=(None, num_passbands, time_series_dim))

    x = input
    x = build_block_stack(x, layers.conv, conv_block, dim=dim)
    if model_info.build.model.startswith('resnet'):
        if layers.res is None:
            raise ValueError(
                'Residual layers must be specified for resnet model.')
        x = build_block_stack(x, layers.res, residual_block, dim=dim)
    x = global_pool_block(x, layers.global_pooling, dim=dim)
    x = tf.keras.layers.Flatten()(x)
    if layers.dense is not None:
        x = build_block_stack(x, layers.dense, dense_block)
    predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=input, outputs=predictions)

    if verbose:
        model.summary()

    return model


def process_X_values(X, model_info: ModelInfo, dim=1):
    X = mean_std(X)
    if dim == 1:
        X = flatten_covariates(X)
    return X
