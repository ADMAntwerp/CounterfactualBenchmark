import random as python_random

import time
import sys
sys.path.append('../frameworks/cadex/')

import tensorflow as tf
from keras.models import Model
from keras import layers
from tensorflow.keras.models import load_model

import pandas as pd
import numpy as np

from Cadex import Cadex
from benchmark.cfg import CFmodel
from benchmark.cfg import OHConverter

from constants.var_types import VAR_TYPES


# Set seeds
np.random.seed(42)
python_random.seed(42)
tf.random.set_random_seed(42)

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

            model_keras.add(tf.keras.layers.Dense(2, activation='softmax'))

            # Additional model layer to input a two inputs instead one binary
            model_keras.layers[5].set_weights([np.array([[-1, 1]]), np.array([1, 0])])


            def insert_intermediate_layer_in_keras(model, layer_id, new_layer):

                layers = [l for l in model.layers]

                x = layers[0].output
                for i in range(1, len(layers)):
                    if i == layer_id:
                        x = new_layer(x)
                    x = layers[i](x)

                # new_model = Model(input=layers[0].input, output=x)
                new_model = Model(layers[0].input, x)
                return new_model

            if cat_feats:
                model_keras_new = insert_intermediate_layer_in_keras(model_keras, 0,
                                                                     layers.Dense(df_oh.shape[1], input_dim=df_oh.shape[1]))
            else:
                model_keras_new = insert_intermediate_layer_in_keras(model_keras, 0,
                                                                     layers.Dense(df.shape[1], input_dim=df.shape[1]))

            cfs_CADEX = []
            if num_feats:
                for idx_cf in range(cf.shape[0]):

                    if cat_feats:
                        converter = OHConverter.Converter(df, cat_feats, list(df_oh.columns))
                        original_point = converter.convert_to_oh(cf.drop(columns=['output']).iloc[idx_cf].tolist())
                        original_point = pd.DataFrame(original_point).T

                        cat_idxs = []
                        for f_idx, c in enumerate(list(df_oh.drop(columns=['output']).columns)):
                            c_idx = int(float(c.split('_')[0]))

                            if len(cat_idxs) == c_idx:
                                cat_idxs.append([])

                            cat_idxs[c_idx].append(f_idx)
                    else:
                        original_point = cf.drop(columns=['output'])[idx_cf: idx_cf+1].copy()

                    cadex_cf = Cadex(model_keras_new, categorical_attributes=cat_idxs if cat_feats else None)

                    start_time = time.time()
                    cfs_out = cadex_cf.train(original_point, 1, 2)

                    if not(cfs_out[0] is None):
                        timeRunCADEX = [idx_cf, int(c), dsName, time.time() - start_time]
                        pd.DataFrame(timeRunCADEX).T.to_csv('../cfoutput/TIME_CADEX.csv', mode='a',
                                                                header=False,
                                                                index=False)

                        cfs_CADEX.append(cfs_out[0][0].tolist())
                        print(dsName, model_keras_new.predict(original_point),
                              model_keras_new.predict(cfs_out[0]))
                    else:
                        cfs_CADEX.append([np.NaN] * original_point.shape[1])
                        print(dsName, 'Not found')

                pd.DataFrame(cfs_CADEX).to_csv(f'../cfoutput/{str(int(c))}_{dsName}_CADEX.csv', index=False)