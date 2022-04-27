import datetime
import signal
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from typing import Optional, Tuple, overload

from marimba.model.io import DataGenerator, from_model_info
from marimba.model.model_info import ModelInfo
from marimba.model.callbacks import checkpoint_callback, tensorboard_callback
from marimba.settings import *
from marimba.model.io import dump_model_info
from numpy.typing import NDArray
import pathlib


@overload
def train(model_info: ModelInfo, *,
          generators: Tuple[DataGenerator, DataGenerator],
          datasets: Literal[None] = None,
          save: bool = True): ...


@overload
def train(model_info: ModelInfo, *,
          datasets: Tuple[NDArray, NDArray,  NDArray, NDArray],
          generators: Literal[None] = None,
          save: bool = True): ...


def train(
        model_info: ModelInfo, *,
        datasets: Optional[Tuple[NDArray, NDArray, NDArray, NDArray]] = None,
        generators: Optional[Tuple[DataGenerator, DataGenerator]] = None,
        save=True):
    """
    Loads a model from a model_info specs, trains it using generators or datasets, and saves it.
    """


    if datasets is None and generators is None:
        raise ValueError("Either datasets or generators must be specified.")
    if datasets is not None and generators is not None:
        raise ValueError(
            "Only one of datasets or generators can be specified.")

    model = from_model_info(model_info)

    current_date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    render_name = f'{current_date}_{model_info.build.model}_{model_info.metadata.name}'

    if save:
        dump_model_info(model_info, render_name)

    if save:
        callbacks = [
            tensorboard_callback(render_name),
            checkpoint_callback(render_name),
        ]
    else:
        callbacks = []
    
    def graceful_termination(_0 = None, _1 = None):
        nonlocal model
        nonlocal render_name
        print('Training interrupted.')
        if save:
            print('Saving model')
            file_path = pathlib.Path(SAVED_MODELS_DIR) / 'fits' / f'{render_name}.h5'
            model.save(file_path)
            print(f'Model saved to {file_path}')

    signal.signal(signal.SIGTERM, graceful_termination)

    try:
        if generators is not None:
            train_generator, validation_generator = generators
            model.fit_generator(
                train_generator,
                steps_per_epoch=len(train_generator),
                epochs=model_info.train.epochs,
                validation_data=validation_generator,
                validation_steps=len(validation_generator),
                callbacks=callbacks,
                workers=model_info.train.workers,
                use_multiprocessing=model_info.train.use_multiprocessing,
                max_queue_size=model_info.train.max_queue_size,
            )
        elif datasets is not None:
            X_train, X_test, y_train, y_test = datasets
            model.fit(
                X_train,
                y_train,
                batch_size=model_info.train.batch_size,
                epochs=model_info.train.epochs,
                validation_data=(X_test, y_test),
                callbacks=callbacks,
            )
    except KeyboardInterrupt:
        graceful_termination()
    


    return model
