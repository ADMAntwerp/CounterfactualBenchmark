import os

import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import load_model

ABS_PATH = '/'.join(str(__file__).split('/')[:-1])


class CreateModel:

    def __init__(self, datasetName, xClass):
        self.model = load_model(f'{ABS_PATH}/../models/{str(xClass)}_{datasetName}.h5')

    def predict(self, x):
        return pd.DataFrame(self.model.predict(x))[0].map(round).to_numpy()

    def predict_proba(self, x):
        return self.model.predict(x)
