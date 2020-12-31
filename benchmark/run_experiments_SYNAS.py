import random as python_random
import gc
import os

import time
import sys
sys.path.append('../frameworks/synth-action-seq/')
sys.path.append('../')

import tensorflow as tf
from tensorflow.keras.models import load_model

import pandas as pd
import numpy as np

from modelssynas import loader
from heuristics.loader import load_heuristics
from recourse.search import SequenceSearch
from recourse.config import base_config

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



                cfs_SYNAS = []
                for idx_cf in range(cf.shape[0]):
                    with tf.Session() as session:
                        # Load model
                        model_keras = load_model(f'../models/{str(int(c))}_{dsName}.h5', compile=False)
                        # model = CFmodel.CreateModel(dsName, str(int(c)))

                        model_keras.add(tf.keras.layers.Dense(2, activation='softmax'))

                        # Additional model layer to input a two inputs instead one binary
                        model_keras.layers[3].set_weights([np.array([[-1, 1]]), np.array([1, 0])])

                        class model_synas:
                            def __init__(self, model_keras):
                                self.FALSE_LABEL = [0.0, 1.0]
                                self.TRUE_LABEL = [1.0, 0.0]
                                self.input_dim = model_keras.layers[0].get_weights()[0].shape[0]
                                self.model = model_keras

                            # def __call__(self, instance_tensor):
                            #     if self.model is None:
                            #         raise ValueError('Model has not been configured')
                            #     return self.model(instance_tensor)


                        model_synas_nn = model_synas(model_keras)

                        if cat_feats:
                            df_train_input = df_oh_train

                            converter = OHConverter.Converter(df, cat_feats, list(df_oh.columns))
                            dict_feat_idx = converter.dict_feat_idx
                            bin_feats = converter.binary_cats

                            factual_input = np.array(converter.convert_to_oh(cf.drop(columns=['output']).iloc[idx_cf].to_list()), dtype='float32')
                        else:
                            df_train_input = df_train
                            dict_feat_idx = []
                            bin_feats = []

                            factual_input = np.array(cf.drop(columns=['output']).iloc[idx_cf].to_list(), dtype='float32')


                        data, actions, features, target_label = loader.setup_generic(df_train_input, cat_feats, num_feats, bin_feats, dict_feat_idx)

                        for name, feature in features.items():
                            feature.initialize_tf_variables()

                        heuristics = load_heuristics('vanilla', actions, model_synas_nn, 1)
                        search = SequenceSearch(model_synas_nn, actions, heuristics, sav_dir=None, config=base_config)

                        print(c, dsName)

                        start_time = time.time()
                        result = search.find_correction(factual_input.reshape((1, factual_input.shape[0])), np.array([target_label]), session)

                        if result.best_result is not None:
                            print(
                                'Final Results',
                                model_keras.predict([[factual_input]]),
                                model_keras.predict([[result.best_result.final_instance]])
                            )

                            timeRunSYNAS = [idx_cf, int(c), dsName, time.time() - start_time]
                            pd.DataFrame(timeRunSYNAS).T.to_csv('../cfoutput/TIME_SYNAS.csv', mode='a',
                                                                header=False,
                                                                index=False)

                            cfs_SYNAS.append(result.best_result.final_instance.tolist())
                        else:
                            cfs_SYNAS.append([np.NaN] * factual_input.shape[0])

                        print(f'#################################### {dsName}: {c}, {idx_cf}')
                    del model_keras, model_synas_nn, df_train_input, dict_feat_idx, \
                        bin_feats, factual_input, data, actions, features, target_label, heuristics, search, \
                        start_time, result
                    if cat_feats:
                        del converter
                    gc.collect()
                    tf.contrib.keras.backend.clear_session()

                pd.DataFrame(cfs_SYNAS).to_csv(f'../cfoutput/{str(int(c))}_{dsName}_SYNAS.csv', index=False)
        except FileNotFoundError:
            print(f'{str(int(c))}_{dsName} has no data')