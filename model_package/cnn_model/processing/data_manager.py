import os
from glob import glob
from pathlib import Path
from typing import List

import pandas as pd
import torch

from cnn_model import __version__ as _version
from cnn_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config
from cnn_model.model import CNN_Train


def load_dataset() -> pd.DataFrame:

    data = []
    for class_folder in os.listdir(DATASET_DIR):
        class_folder_dir = os.path.join(DATASET_DIR, class_folder)
        # print(glob(os.path.join(class_folder_dir, "*.png")))
        for img in glob(os.path.join(class_folder_dir, "*.png")):
            r = [img, class_folder]
            data.append(r)
    # print(data)
    df = pd.DataFrame(data, columns=config.model_conf.data_columns)

    return df


def save_model(model: CNN_Train, epoch: int, optimizer: torch.optim.Adam, loss: float):

    model_name = f"{config.app_config.model_save_file}{_version}.pth".format(epoch)
    model_path = Path(f"{TRAINED_MODEL_DIR}/{model_name}")

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        model_path,
    )


def load_model(epoch: int):

    model_name = f"{config.app_config.model_save_file}{_version}.pth".format(epoch)
    model_path = Path(f"{TRAINED_MODEL_DIR}/{model_name}")

    checkpoint = torch.load(model_path)

    return checkpoint


def delete_model(epochs: List[int]):

    for epoch in epochs:
        model_name = f"{config.app_config.model_save_file}{_version}.pth".format(epoch)
        model_path = Path(f"{TRAINED_MODEL_DIR}/{model_name}")

        os.remove(model_path)
