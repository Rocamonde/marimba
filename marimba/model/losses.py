import numpy as np
import tensorflow as tf

from marimba.model.model_info import ModelInfo


def class_weighted_crossentropy(model_info: ModelInfo):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Usage:
        loss = weighted_categorical_crossentropy()
        model.compile(loss=loss,optimizer='adam')
    """

    def class_weighted_crossentropy_loss(y_true, y_pred):

        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= tf.reduce_sum(y_pred, axis=-1, keepdims=True)

        # clip to prevent NaN's and Inf's
        y_pred = tf.clip_by_value(
            y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

        vanilla_loss = y_true * tf.math.log(y_pred)

        loss_by_class = tf.reduce_sum(vanilla_loss, axis=0)

        count_by_class = tf.reduce_sum(y_true, axis=0)
        count_by_class = tf.clip_by_value(
            count_by_class,
            clip_value_min=tf.keras.backend.epsilon(),
            clip_value_max=np.inf)

        # just use reduce_mean
        loss = -tf.reduce_mean(loss_by_class / count_by_class)

        loss = loss

        return loss

    return class_weighted_crossentropy_loss


losses_mapping = {
    'class_weighted_crossentropy': class_weighted_crossentropy
}


def get_loss(name: str):
    return losses_mapping[name]
