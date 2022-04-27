import os
import pathlib
from typing import Tuple

import h5py
import pandas as pd
import yaml
from marimba.model.losses import *
from marimba.model.metrics import *
from marimba.model.model_info import ModelInfo, load_model_info
from marimba.settings import *
from numpy.typing import NDArray
from marimba.model.conv1d import build_model, process_X_values
from marimba.model.resnet import build_model as build_resnet, preprocess as preprocess_resnet


models_dict = {
    'conv1d': {
        'build': build_model,
        'preprocess': process_X_values,
    },
    'conv2d': {
        'build': build_model,
        'preprocess': process_X_values,
    },
    'resnet1d': {
        'build': build_model,
        'preprocess': process_X_values,
    },
    'resnet2d': {
        'build': build_resnet,
        'preprocess': preprocess_resnet,
    }
}


class DataGenerator(tf.keras.utils.Sequence):  # type: ignore
    def __init__(self, file_path, key, indices: NDArray, labels: NDArray, batch_size: int):
        self.file_path = file_path
        self.labels = labels
        self.batch_size = batch_size
        self.key = key
        self.indices = indices

    def __len__(self):
        return int(np.ceil(len(self.labels) / float(self.batch_size)))

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as hf:
            batch_indices = self.indices[idx *
                                         self.batch_size:(idx + 1) * self.batch_size]
            ascending_batch_indices = batch_indices[np.argsort(batch_indices)]
            X = hf.get(self.key)[ascending_batch_indices]
            X = self.preprocess(X)
            y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        return X, y

    def preprocess(self, X):
        """
        Preprocess the data. This is a no-op by default.
        Can be overridden by subclasses.
        """
        return X


def compile_model(model, model_info: ModelInfo):
    """
    Compiles the model. Given the model_info object, it will select the loss, optimizer, 
    learning rate, and metrics to be tracked.
    """
    loss = model_info.compile.loss
    learning_rate = model_info.compile.learning_rate
    if loss.startswith('marimba.'):
        loss_fn = get_loss(loss.lstrip('marimba.'))(model_info)
    else:
        loss_fn = loss

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=['categorical_accuracy', class_weighted_categorical_accuracy()])
    return model


def dump_model_info(model_info: ModelInfo, filename: str) -> None:
    """
    Dump the model info to a file.
    """
    log_dir = pathlib.Path(SAVED_MODELS_DIR) / 'model_info'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    file_path = log_dir / f'{filename}.yaml'
    with open(file_path, "w") as f:
        yaml.dump(model_info.dict(), f)


def from_model_info(model_info: ModelInfo):
    """
    Load a model from a model_info object.
    """
    model = models_dict[model_info.build.model]['build'](model_info)
    compile_model(model, model_info)
    return model


def from_yaml(model_info_path: str):
    """
    Load a model from a yaml file.
    """
    model_info = load_model_info(model_info_path)
    model = from_model_info(model_info)
    return model_info, model


def from_hdf5(model_h5_path: str):
    """
    Load a model from a hdf5 file.
    """
    if model_h5_path.endswith('.h5') or model_h5_path.endswith('.hdf5'):
        file_path =  pathlib.Path(SAVED_MODELS_DIR) / model_h5_path
    else:
        file_path =  pathlib.Path(SAVED_MODELS_DIR) / f'{model_h5_path}.h5'
    model = tf.keras.models.load_model(file_path, custom_objects={
        'class_weighted_categorical_accuracy': class_weighted_categorical_accuracy,
        'class_weighted_crossentropy_loss': class_weighted_crossentropy,
    })
    return model


def load_metadata(name: str):
    """
    Load the metadata for a given dataset. This contains the object information, 
    including the class labels, object IDs, and other metadata covariates.
    """
    data_path = pathlib.Path(DATA_DIR) / f'{name}.h5'
    store = pd.HDFStore(data_path, "r")
    dataframe = pd.read_hdf(store, 'metadata')
    dataframe.sort_index(inplace=True)
    return dataframe


# def get_dataset_generator(name: str, batch_size, generator_cls: Optional[Type[DataGenerator]] = None, test_size=0.2) -> Tuple[DataGenerator, DataGenerator]:
#     metadata = load_metadata(name)
#     fits_path = os.path.join(DATA_FITS_DIR, 'fit_' + name + ".h5")
#     y = np.array(metadata['class'])
#     old_labels, y = np.unique(y, return_inverse=True)
#     y = np.array(tf.one_hot(y, len(old_labels)))
#     indices = np.arange(len(y))
#     X_idx_train, X_idx_test, y_train, y_test = train_test_split(
#         indices, y, test_size=test_size, random_state=42)
#     if generator_cls is None:
#         generator_cls = Generator

#     train_generator = generator_cls(
#         file_path=fits_path, key=FIT_VERSION, indices=X_idx_train, labels=y_train, batch_size=batch_size)
#     test_generator = generator_cls(
#         file_path=fits_path, key=FIT_VERSION, indices=X_idx_test, labels=y_test, batch_size=batch_size)
#     return train_generator, test_generator


def get_dataset(name: str, slice=None, one_hot=True) -> Tuple[NDArray, NDArray]:
    """
    Loads a dataset of fitted light curves, and returns the data and labels.
    """
    # fits_path using pathlib
    fits_path = pathlib.Path(DATA_FITS_DIR) / f'fit_{name}.h5'
    metadata = load_metadata(name)

    if slice is not None:
        metadata = metadata.iloc[:slice]

    y = np.array(metadata['class'])

    with h5py.File(fits_path, 'r') as hf:
        if slice is None:
            X = hf.get(FIT_VERSION)[:]
        else:
            X = hf.get(FIT_VERSION)[:slice]

    # TODO this is a temporary fix
    # exclude the corresponding rows from X and y (these are the unseen classes)
    X = X[y < 99]
    y = y[y < 99]

    old_labels, y = np.unique(y, return_inverse=True)

    if one_hot:
        y = np.array(tf.one_hot(y, len(old_labels)))

    return X, y
