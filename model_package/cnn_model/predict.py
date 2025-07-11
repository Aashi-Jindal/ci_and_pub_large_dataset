from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from cnn_model.config.core import TRAINED_MODEL_DIR, config
from cnn_model.model import CNN_Train
from cnn_model.processing.data_manager import load_model
from cnn_model.processing.preprocessing import CreateDataset


def predict_label(Xtest: pd.DataFrame, epoch: int) -> List[float]:

    create_dataset = CreateDataset()
    Xtest = create_dataset.transform(Xtest)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xtest_tensor = (torch.tensor(Xtest, dtype=torch.float32)).permute(0, 3, 1, 2)

    checkpoint = load_model(epoch)

    model = CNN_Train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.model_conf.lr)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    test_loader = DataLoader(
        Xtest_tensor, batch_size=config.model_conf.batch_size, shuffle=True
    )

    model.eval()
    predictions = []

    with torch.no_grad():
        for img in test_loader:
            img = img.to(device)

            out = model(img)
            _, predicted = out.max(1)
            predicted = [float(a) for a in predicted]
            predictions.extend(predicted)

    return predictions


if __name__ == "__main__":

    from sklearn.model_selection import train_test_split

    from cnn_model.processing.data_manager import load_dataset

    data = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_conf.data_columns[0]],
        data[config.model_conf.data_columns[1]],
        test_size=config.model_conf.test_size,
        random_state=config.model_conf.random_state,
        shuffle=True,
    )

    X_test.reset_index(drop=True, inplace=True)

    predictions = predict_label(X_test, 30)

    lm_path = Path(f"{TRAINED_MODEL_DIR}/{config.model_conf.target_encoder_name}")
    lm = joblib.load(lm_path)
    y_test = lm.transform(y_test)
    conf_mat = confusion_matrix(y_test, np.array(predictions))

    print(conf_mat)
