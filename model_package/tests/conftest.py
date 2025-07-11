import pytest
from sklearn.model_selection import train_test_split

from cnn_model.config.core import config
from cnn_model.processing.data_manager import load_dataset


@pytest.fixture
def sample_input_data():

    data = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_conf.data_columns[0]],
        data[config.model_conf.data_columns[1]],
        test_size=config.model_conf.test_size,
        random_state=config.model_conf.random_state,
        shuffle=True,
    )

    X_test.reset_index(drop=True, inplace=True)

    return X_test, y_test
