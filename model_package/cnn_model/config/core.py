from pathlib import Path
from typing import List

from pydantic import BaseModel
from strictyaml import load

import cnn_model

PACKAGE_ROOT = Path(cnn_model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = PACKAGE_ROOT / "datasets" / "v2-plant-seedlings-dataset"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"


class AppConfig(BaseModel):
    """
    Application-level config.
    """

    package_name: str
    model_name: str
    model_save_file: str


class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training.
    """

    target_encoder_name: str
    IMAGE_SIZE: int
    n_classes: int
    data_columns: List[str]
    test_size: float
    random_state: int
    batch_size: int
    kernel_size: int
    out_channel1: int
    out_channel2: int
    out_channel3: int
    dropout_p: float
    dense_neurons: int
    lr: float
    epochs: int


class Config(BaseModel):
    """Master config object."""

    app_config: AppConfig
    model_conf: ModelConfig


def create_and_validate_config() -> Config:
    """Run validation on config values."""

    # CONFIG_FILE_PATH = "regression_model/config.yml"

    with open(CONFIG_FILE_PATH, "r") as conf_file:
        parsed_config = load(conf_file.read())
    # print(parsed_config.data)

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        app_config=AppConfig(**parsed_config.data),
        model_conf=ModelConfig(**parsed_config.data),
    )

    return _config


config = create_and_validate_config()
