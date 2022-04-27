import tensorflow as tf
from marimba.model.model_info import (ConvLayer, DenseLayer,
                                      GlobalPoolingLayer)
from marimba.model.regularization import get_regularizers

def conv_block(x, conv_layer: ConvLayer, dim=1):
    """
    Builds a convolutional layer block.
    """
    if dim != 1 and dim != 2:
        raise ValueError(f'Dimension {dim} not supported')

    regularizers = get_regularizers(conv_layer.regularizers)
    convolution = tf.keras.layers.Conv1D if dim == 1 else tf.keras.layers.Conv2D
    max_pool = tf.keras.layers.MaxPooling1D if dim == 1 else tf.keras.layers.MaxPooling2D
    avg_pool = tf.keras.layers.AveragePooling1D if dim == 1 else tf.keras.layers.AveragePooling2D

    x = convolution(
        filters=conv_layer.filters,
        kernel_size=conv_layer.kernel_size,
        strides=conv_layer.strides,
        padding=conv_layer.padding,
        **regularizers
    )(x)

    if conv_layer.dropout > 0:
        x = tf.keras.layers.Dropout(conv_layer.dropout)(x)

    if conv_layer.batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)

    x = getattr(tf.nn, conv_layer.activation)(x)

    if conv_layer.pooling:
        if conv_layer.pooling.type == 'max':
            x = max_pool(pool_size=conv_layer.pooling.size)(x)
        elif conv_layer.pooling.type == 'avg':
            x = avg_pool(pool_size=conv_layer.pooling.size)(x)
        else:
            raise ValueError(
                f'Pooling type {conv_layer.pooling.type} not supported')
    return x


def dense_block(x, dense_layer: DenseLayer, **kwargs):
    """
    Builds a dense layer block.
    """
    regularizers = get_regularizers(dense_layer.regularizers)
    x = tf.keras.layers.Dense(
        units=dense_layer.units,
        activation=dense_layer.activation,
        **regularizers
    )(x)

    if dense_layer.dropout > 0:
        x = tf.keras.layers.Dropout(dense_layer.dropout)(x)

    return x


def global_pool_block(x, global_pooling_layer: GlobalPoolingLayer, dim=1):
    """
    Builds a global pooling layer block.
    """
    if dim != 1 and dim != 2:
        raise ValueError(f'Dimension {dim} not supported')

    if global_pooling_layer.type == 'max':
        max_pool = tf.keras.layers.GlobalMaxPool1D if dim == 1 else tf.keras.layers.GlobalMaxPool2D
        x = max_pool()(x)
    elif global_pooling_layer.type == 'avg':
        avg_pool = tf.keras.layers.GlobalAveragePooling1D if dim == 1 else tf.keras.layers.GlobalAveragePooling2D
        x = avg_pool()(x)
    else:
        raise ValueError(
            f'Pooling type {global_pooling_layer.type} not supported')

    if global_pooling_layer.dropout > 0:
        x = tf.keras.layers.Dropout(global_pooling_layer.dropout)(x)

    return x


def residual_block(x, conv_layer_info: ConvLayer, dim=1):
    """
    Creates a residual block for a given input vector x
    """
    conv_1 = conv_block(x, conv_layer=conv_layer_info, dim=dim)
    conv_2 = conv_block(conv_1, conv_layer=conv_layer_info, dim=dim)
    return tf.keras.layers.Add()([x, conv_2])

