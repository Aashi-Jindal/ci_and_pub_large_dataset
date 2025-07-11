import cv2
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

from cnn_model.config.core import config


class LabelMapping(BaseEstimator, TransformerMixin):

    def __init__(self):

        self.encoder: LabelEncoder = LabelEncoder()

    def fit(self, X: pd.DataFrame, y: pd.Series = None):

        self.encoder.fit_transform(X)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        X_trans = self.encoder.transform(X)

        return X_trans


class CreateDataset(BaseEstimator, TransformerMixin):

    def __init__(self):

        self.image_size = config.model_conf.IMAGE_SIZE

    def fit(self, X: pd.DataFrame, y: pd.Series = None):

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:

        data = []  # n1, n2, n3, n4
        for i, _ in enumerate(X):
            im = cv2.imread(X.loc[i])
            im = cv2.resize(im, [self.image_size, self.image_size])
            data.append(im)

        data = np.array(data)

        return data
