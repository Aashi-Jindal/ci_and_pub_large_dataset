import torch
import torch.nn as nn

from cnn_model.config.core import config


class CNN_Train(nn.Module):

    def __init__(self):
        super().__init__()

        self.cnn_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=config.model_conf.out_channel1,
                kernel_size=config.model_conf.kernel_size,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=config.model_conf.out_channel1,
                out_channels=config.model_conf.out_channel1,
                kernel_size=config.model_conf.kernel_size,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=config.model_conf.kernel_size, stride=2),
            nn.Dropout(p=config.model_conf.dropout_p),
            nn.Conv2d(
                in_channels=config.model_conf.out_channel1,
                out_channels=config.model_conf.out_channel2,
                kernel_size=config.model_conf.kernel_size,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=config.model_conf.out_channel2,
                out_channels=config.model_conf.out_channel2,
                kernel_size=config.model_conf.kernel_size,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=config.model_conf.kernel_size, stride=2),
            nn.Dropout(p=config.model_conf.dropout_p),
            nn.Conv2d(
                in_channels=config.model_conf.out_channel2,
                out_channels=config.model_conf.out_channel3,
                kernel_size=config.model_conf.kernel_size,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=config.model_conf.out_channel3,
                out_channels=config.model_conf.out_channel3,
                kernel_size=config.model_conf.kernel_size,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=config.model_conf.kernel_size, stride=2),
            nn.Dropout(p=config.model_conf.dropout_p),
        )

        with torch.no_grad():
            dummy_input = torch.randn(
                1, 3, config.model_conf.IMAGE_SIZE, config.model_conf.IMAGE_SIZE
            )
            dummy_output = self.cnn_layer(dummy_input)
            flattened_size = dummy_output.view(1, -1).size(1)

        self.fc1 = nn.Linear(
            in_features=flattened_size, out_features=config.model_conf.dense_neurons
        )
        self.dropout = nn.Dropout(p=config.model_conf.dropout_p)
        self.fc2 = nn.Linear(
            in_features=config.model_conf.dense_neurons,
            out_features=config.model_conf.n_classes,
        )

    def forward(self, x):
        x = self.cnn_layer(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


cnn_model = CNN_Train()
