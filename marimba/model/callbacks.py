import tensorflow as tf
from marimba.settings import *
import pathlib


def tensorboard_callback(name: str):
    """
    This callback allows registering a per-epoch log of the training and validation performance of the model, 
    for later analysis, model comparison and visualization using the TensorBoard front-end.
    """
    log_dir = pathlib.Path(LOGS_DIR) / 'fits' / name
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=str(log_dir), histogram_freq=1)
    return tensorboard_callback


def checkpoint_callback(name: str):
    """
    This callback allows saving the model after each epoch, so that models can be trained without interruptions
    nor worrying about the number of epochs or of overfitting.
    """
    checkpoint_dir = pathlib.Path(
        SAVED_MODELS_DIR) / 'fits' / 'checkpoints' / name / "{epoch:06d}.hdf5"
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        str(checkpoint_dir), save_weights_only=False, verbose=1, save_freq="epoch")
    return model_checkpoint
