import tensorflow as tf


def class_weighted_categorical_accuracy():
    """
    A weighted version of keras.metrics.categorical_accuracy

    Usage:
        acc = weighted_categorical_accuracy()
        model.compile(loss=loss,optimizer='adam',metrics=[acc])
    """

    def class_weighted_categorical_accuracy(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= tf.reduce_sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = tf.clip_by_value(
            y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        # calc
        y_pred_sparse = tf.argmax(y_pred, axis=-1)
        # convert y_pred_sparse to dense tensor
        y_pred_sparse = tf.cast(y_pred_sparse, tf.int32)
        y_pred_sparse = tf.cast(tf.one_hot(
            y_pred_sparse, y_pred.shape[-1]), tf.int32)

        y_true_sparse = tf.cast(y_true, tf.int32)
        # obtain the number of correct predictions
        matches = tf.multiply(y_true_sparse, y_pred_sparse)
        matches_by_class = tf.cast(tf.reduce_sum(matches, axis=0), tf.float32)
        count_by_class = tf.cast(tf.reduce_sum(
            y_true_sparse, axis=0), tf.float32)
        # we perform a safe divide so that if the count is zero, we return zero
        accuracy_by_class = tf.math.divide_no_nan(
            matches_by_class, count_by_class)
        accuracy = tf.reduce_mean(accuracy_by_class)
        return accuracy

    return class_weighted_categorical_accuracy


metrics_mapping = {
    'class_weighted_categorical_accuracy': class_weighted_categorical_accuracy,
}


def get_metric(name):
    return metrics_mapping[name]
