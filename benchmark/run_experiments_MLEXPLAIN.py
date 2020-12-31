import random as python_random

import time
import sys
import gc
sys.path.append('../frameworks/ml-explain/')
sys.path.append('../')

import tensorflow as tf
from tensorflow.keras.models import load_model
import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np

from mlexplain.sce.tabular_sce import TabularSCE
from mlexplain.counterfactuals.vanilla_gradient_counterfactual import VanillaGradientCounterfactual
from benchmark.cfg import CFmodel
from benchmark.cfg import OHConverter

from constants.var_types import VAR_TYPES


# Set seeds
np.random.seed(42)
python_random.seed(42)
tf.random.set_seed(42)
tf.compat.v1.disable_v2_behavior() # disable TF2 behaviour as alibi code still relies on TF1 constructs

print('TF version: ', tf.__version__)
print('Eager execution enabled: ', tf.executing_eagerly()) # False


for dsName in VAR_TYPES.keys():

    # Load feature type specifications
    cat_feats = VAR_TYPES[dsName]['categorical']
    num_feats = VAR_TYPES[dsName]['numerical']


    # Load Dataset
    if cat_feats and num_feats:
        df = pd.read_csv(f'../data/NORM_{dsName}.csv')
        df_oh = pd.read_csv(f'../data/OH_NORM_{dsName}.csv')
    elif cat_feats:
        df = pd.read_csv(f'../data/{dsName}.csv')
        df_oh = pd.read_csv(f'../data/OH_{dsName}.csv')
    else:
        df = pd.read_csv(f'../data/NORM_{dsName}.csv')
        df_oh = []

    df_y_original = df['output'].copy()
    df_oh_y_original = df['output'].copy()

    # Iterate over the classes
    for c in list(df['output'].unique()):

        try:
            cf = pd.read_csv(f'../experiments_data/{str(int(c))}_{dsName}_CFDATASET.csv').drop(columns=['Unnamed: 0'])

            # If there are CF to be analyzed
            if cf.shape[0] > 0:

                # Load train data
                df_train = pd.read_csv(f'../experiments_data/{str(int(c))}_{dsName}_TRAINDATASET.csv').drop(columns=['Unnamed: 0'])
                # Load test data
                df_test = pd.read_csv(f'../experiments_data/{str(int(c))}_{dsName}_TESTDATASET.csv').drop(columns=['Unnamed: 0'])
                # Load OH if existent
                if cat_feats:
                    # Load train data
                    df_oh_train = pd.read_csv(f'../experiments_data/{str(int(c))}_{dsName}_TRAINOHDATASET.csv').drop(columns=['Unnamed: 0'])
                    # Load test data
                    df_oh_test = pd.read_csv(f'../experiments_data/{str(int(c))}_{dsName}_TESTOHDATASET.csv').drop(columns=['Unnamed: 0'])

                # Load model
                model_keras = load_model(f'../models/{str(int(c))}_{dsName}.h5', compile=False)
                model = CFmodel.CreateModel(dsName, str(int(c)))

                cfs_MLEXPLAIN = []
                for idx_cf in range(cf.shape[0]):
                    class SimpleNN(nn.Module):
                        def __init__(self):
                            super().__init__()

                            shapeL1 = model_keras.layers[0].get_weights()[0].shape
                            self.layer1 = nn.Linear(shapeL1[0], shapeL1[1])

                            shapeL2 = model_keras.layers[1].get_weights()[0].shape
                            self.layer2 = nn.Linear(shapeL2[0], shapeL2[1])

                            shapeLO = model_keras.layers[2].get_weights()[0].shape
                            self.layerO = nn.Linear(shapeLO[0], shapeLO[1])

                        def forward(self, x):
                            """ Accepts input of shape (N_objects, n_input) """
                            x = self.layer1(x)
                            x = torch.relu(x)
                            x = self.layer2(x)
                            x = torch.relu(x)
                            x = self.layerO(x)
                            x = torch.sigmoid(x)
                            return x


                    classifier = SimpleNN()

                    classifier.layer1.weight.data = torch.tensor(model_keras.layers[0].get_weights()[0].T)
                    classifier.layer1.bias.data = torch.tensor(model_keras.layers[0].get_weights()[1])

                    classifier.layer2.weight.data = torch.tensor(model_keras.layers[1].get_weights()[0].T)
                    classifier.layer2.bias.data = torch.tensor(model_keras.layers[1].get_weights()[1])

                    classifier.layerO.weight.data = torch.tensor(model_keras.layers[2].get_weights()[0].T)
                    classifier.layerO.bias.data = torch.tensor(model_keras.layers[2].get_weights()[1])

                    if cat_feats:
                        converter = OHConverter.Converter(df, cat_feats, list(df_oh.columns))
                        cf_origin = converter.convert_to_oh(cf.drop(columns=['output']).iloc[idx_cf].to_list())
                    else:
                        cf_origin = cf.drop(columns=['output']).iloc[idx_cf:idx_cf+1].to_numpy()

                    cfg = VanillaGradientCounterfactual(classifier)

                    print(c, dsName)
                    start_time = time.time()
                    cf_output = cfg.generate(torch.tensor([cf_origin], dtype=torch.float32), [[0]])
                    timeRunMLEXPLAIN = [idx_cf, int(c), dsName, time.time() - start_time]

                    pd.DataFrame(timeRunMLEXPLAIN).T.to_csv('../cfoutput/TIME_MLEXPLAIN.csv', mode='a', header=False,
                                                            index=False)

                    cf_out_array = cf_output.detach().numpy().tolist()
                    if len(cf_out_array[0]) > 1:
                        cfs_MLEXPLAIN.append(cf_out_array[0])
                    else:
                        cfs_MLEXPLAIN.append(cf_out_array[0][0])

                    gc.collect()

                pd.DataFrame(cfs_MLEXPLAIN).to_csv(f'../cfoutput/{str(int(c))}_{dsName}_MLEXPLAIN.csv', index=False)
        except FileNotFoundError:
            print(f'{str(int(c))}_{dsName} has no data')