import os
from glob import glob
from pathlib import Path

import joblib
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from cnn_model.config.core import TRAINED_MODEL_DIR, config
from cnn_model.model import CNN_Train
from cnn_model.processing.data_manager import (
    delete_model,
    load_dataset,
    load_model,
    save_model,
)
from cnn_model.processing.preprocessing import CreateDataset, LabelMapping
from cnn_model.processing.utils import fetch_epoch_id, set_seed

set_seed()


def run_training() -> None:

    data = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_conf.data_columns[0]],
        data[config.model_conf.data_columns[1]],
        test_size=config.model_conf.test_size,
        random_state=config.model_conf.random_state,
        shuffle=True,
    )

    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)

    create_dataset = CreateDataset()
    X_train_data = create_dataset.transform(X_train)
    X_test_data = create_dataset.transform(X_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_tensor = (
        torch.tensor(X_train_data[:100, :, :], dtype=torch.float32)
    ).permute(0, 3, 1, 2)
    X_test_tensor = (
        torch.tensor(X_test_data[:100, :, :], dtype=torch.float32)
    ).permute(0, 3, 1, 2)

    lm = LabelMapping()
    lm.fit(y_train)
    y_train = lm.transform(y_train)
    y_test = lm.transform(y_test)

    target_save_file = Path(
        f"{TRAINED_MODEL_DIR}/{config.model_conf.target_encoder_name}"
    )
    joblib.dump(lm, target_save_file)

    y_train_tensor = torch.tensor(y_train[:100], dtype=torch.long)
    y_test_tensor = torch.tensor(y_test[:100], dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(
        train_dataset, batch_size=config.model_conf.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.model_conf.batch_size, shuffle=True
    )

    highest_epoch = 0
    model_fnames = glob(os.path.join(TRAINED_MODEL_DIR, "*.pth"))

    if len(model_fnames) > 0:

        highest_epoch, all_epochs = fetch_epoch_id(model_fnames)

        checkpoint = load_model(highest_epoch)

        model = CNN_Train()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.model_conf.lr)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        delete_model(all_epochs)

    else:
        model = CNN_Train()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.model_conf.lr)

    criterion = nn.CrossEntropyLoss()

    num_epochs = config.model_conf.epochs

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for img, label in train_loader:
            img, label = img.to(device), label.to(device)

            out = model(img)
            loss = criterion(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss
            _, predicted = out.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()

        print(
            f"""Epoch {epoch+1}/{num_epochs},
            Loss: {running_loss/len(train_loader):.4f},
            Accuracy: {100*correct/total:.2f}%"""
        )

    save_model(model, highest_epoch + epoch + 1, optimizer, loss)

    model.eval()

    total = 0
    correct = 0

    with torch.no_grad():
        for img, label in test_loader:
            img, label = img.to(device), label.to(device)

            out = model(img)
            _, predicted = out.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()

    print(f"Test set Accuracy: {100*correct/total:.2f}%")


if __name__ == "__main__":

    run_training()
