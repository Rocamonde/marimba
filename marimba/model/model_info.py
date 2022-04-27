try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from typing import List, Optional, Union
from pydantic import BaseModel
import yaml


class GlobalPoolingLayer(BaseModel):
    type: Union[Literal['max'], Literal['avg']]
    dropout: float


class Regularizer(BaseModel):
    l1: float
    l2: float


class Regularizers(BaseModel):
    kernel: Optional[Regularizer] = None
    bias: Optional[Regularizer] = None
    activity: Optional[Regularizer] = None


class DenseLayer(BaseModel):
    number: Optional[int] = None
    units: int
    activation: str
    dropout: float
    regularizers: Optional[Regularizers] = None


class PoolingLayer(BaseModel):
    type: Union[Literal['max'], Literal['avg']]
    size: int


class ConvLayer(BaseModel):
    number: Optional[int] = None
    filters: int
    kernel_size: int
    strides: int
    padding: Union[Literal['same'], Literal['valid']]
    activation: str
    dropout: float
    batch_norm: bool
    pooling: Optional[PoolingLayer] = None
    regularizers: Optional[Regularizers] = None


class Layers(BaseModel):
    conv: Optional[Union[ConvLayer, List[ConvLayer]]]
    res: Optional[Union[ConvLayer, List[ConvLayer]]]
    global_pooling: GlobalPoolingLayer
    dense: Optional[Union[DenseLayer, List[DenseLayer]]]


class Build(BaseModel):
    model: Union[Literal['conv1d'], Literal['conv2d'],
                 Literal['resnet1d'], Literal['resnet2d']]
    layers: Layers
    num_classes: int
    num_passbands: int
    time_series_dim: int


class Compile(BaseModel):
    learning_rate: float
    loss: str


class Train(BaseModel):
    epochs: int
    batch_size: int
    workers: int = 1
    use_multiprocessing: bool = False
    max_queue_size: int = 10


class Metadata(BaseModel):
    name: str
    description: str


class ModelInfo(BaseModel):
    build: Build
    compile: Compile
    train: Train
    metadata: Metadata


def load_model_info(model_info_path: str) -> ModelInfo:
    """
    Load a model info file.

    Parameters
    ----------
    model_info_path : str
        Path to the model info file.

    Returns
    -------
    ModelInfo
        The model info object.
    """
    with open(model_info_path, 'r') as f:
        return ModelInfo.parse_obj(yaml.load(f, Loader=yaml.SafeLoader))
