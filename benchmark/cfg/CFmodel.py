import random as python_random
import os

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model

# Set seeds
np.random.seed(42)
python_random.seed(42)
try:
    tf.random.set_random_seed(42)
except AttributeError:
    tf.random.set_seed(42)

# MUST CHECK!!!!!! For reproducibility
class CreateModel:

    def __init__(self, datasetName, xClass):
        self.model = load_model(f'../models/{str(xClass)}_{datasetName}.h5', compile=False)

    def predict(self, x):
        return pd.DataFrame(self.model.predict(x))[0].map(round).to_numpy()

    def predict_proba(self, x):
        return self.model.predict(x)
