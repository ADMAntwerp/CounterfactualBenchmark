import os

import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import load_model

class CreateModel:

    def __init__(self, datasetName, xClass):
        self.model = load_model(f'../models/{str(xClass)}_{datasetName}.h5')

    def predict(self, x):
        return pd.DataFrame(self.model.predict(x))[0].map(round).to_numpy()

    def predict_proba(self, x):
        return self.model.predict(x)
