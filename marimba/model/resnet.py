from model.model_info import ConvLayer, ModelInfo
from model.regularization import get_regularizers
import tensorflow as tf
from marimba.model.utils import global_pool_block


def build_residual_block(x_l, conv_layer_info: ConvLayer):
    x = x_l
    # convolution
    x = tf.keras.layers.Conv2D(
        filters=conv_layer_info.filters,
        kernel_size=conv_layer_info.kernel_size,
        strides=conv_layer_info.strides,
        padding='same',
        **get_regularizers(conv_layer_info.regularizers)
    )(x)
    # batch normalization
    x = tf.keras.layers.BatchNormalization()(x)
    # activation
    x = tf.keras.layers.Activation(conv_layer_info.activation)(x)


    # convolution
    x = tf.keras.layers.Conv2D(
        filters=conv_layer_info.filters,
        kernel_size=conv_layer_info.kernel_size,
        strides=conv_layer_info.strides,
        padding='same',
        **get_regularizers(conv_layer_info.regularizers)
    )(x)
    # batch normalization
    x = tf.keras.layers.BatchNormalization()(x)

    # 1x1 conv to match dimensions
    x_l = tf.keras.layers.Conv2D(
        filters=conv_layer_info.filters,
        kernel_size=1,
        strides=1,
        padding='same',
        # kernel_initializer='ones',
        **get_regularizers(conv_layer_info.regularizers)
    )(x_l)

    # add residual
    x = tf.keras.layers.Add()([x, x_l])
    # activation
    x = tf.keras.layers.Activation(conv_layer_info.activation)(x)

    # add dropout
    if conv_layer_info.dropout is not None and conv_layer_info.dropout > 0:
        x = tf.keras.layers.Dropout(conv_layer_info.dropout)(x)

    return x


def build_model(model_info: ModelInfo, verbose=False):
    if model_info.build.model not in ['resnet2d']:
        raise ValueError(f'Model {model_info.build.model} not supported')

    num_passbands = model_info.build.num_passbands
    time_series_dim = model_info.build.time_series_dim
    num_classes = model_info.build.num_classes
    res_layers = model_info.build.layers.res

    input = tf.keras.layers.Input(
        shape=(None, num_passbands, time_series_dim))

    x = input
    # pre batch normalization
    x = tf.keras.layers.BatchNormalization()(x)
    if res_layers is None:
        raise ValueError(
            'Residual layers must be specified for resnet model.')
    elif isinstance(res_layers, list):
        for layer in res_layers:
            x = build_residual_block(x, layer)
    else:
        number = res_layers.number
        if number is None:
            raise ValueError(
                'number of layers must be specified, or provide a list of layers.')
        for _ in range(number):
            x = build_residual_block(x, res_layers)

    # global pooling
    x = global_pool_block(x, model_info.build.layers.global_pooling, dim=2)

    # dense layer
    x = tf.keras.layers.Dense(
        units=num_classes,
        activation='softmax',
    )(x)

    model = tf.keras.Model(inputs=input, outputs=x)

    if verbose:
        model.summary()

    return model


def preprocess(X, *args, **kwargs):
    return X
