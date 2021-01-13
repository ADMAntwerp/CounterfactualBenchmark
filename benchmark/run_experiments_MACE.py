import random as python_random

import gc
import time
import sys
from importlib import reload
sys.path.append('../frameworks/mace/')
sys.path.append('../')

import tensorflow as tf
from tensorflow.keras.models import load_model

from sklearn.neural_network import MLPClassifier

import pandas as pd
import numpy as np

from benchmark.cfg import CFmodel
from benchmark.cfg import OHConverter
import generateSATExplanations
from loadData import loadDataset

from constants.var_types import VAR_TYPES


# Set seeds
np.random.seed(42)
python_random.seed(42)
tf.random.set_seed(42)

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
                # model = CFmodel.CreateModel(dsName, str(int(c)))

                cfs_SEDC = []
                for idx_cf in range(cf.shape[0]):
                    generateSATExplanations = reload(generateSATExplanations)

                    # ds = loadDataset('generic', return_one_hot=False, df=df_train, df_oh=None, cat_feats=cat_feats, num_feats=num_feats, bin_feats=[])

                    if cat_feats:
                        converter = OHConverter.Converter(df, cat_feats, list(df_oh.columns))
                        bin_feats = converter.binary_cats

                    else:
                        bin_feats = []

                    df_input = pd.concat([df_train, cf])

                    if cat_feats:
                        for c_feat in list(set(cat_feats) - set(bin_feats)):
                            dict_conversion = {}
                            for idx_cat_key, key_ordered_n in enumerate(converter.dict_feat_idx[c_feat].keys()):
                                if df_input[c_feat].dtype == np.int64:
                                    dict_conversion[int(float(key_ordered_n))] = idx_cat_key + 1
                                else:
                                    dict_conversion[key_ordered_n] = idx_cat_key + 1
                            df_input[c_feat] = df_input[c_feat].map(dict_conversion)


                    df_input = pd.concat([df_input[['output']], df_input.drop(columns=['output'])], axis=1)
                    ds = loadDataset('generic', return_one_hot=False if not cat_feats else True, df=df_input, df_oh=None,
                                     cat_feats=cat_feats, num_feats=num_feats, bin_feats=bin_feats)
                    ds.data_frame_kurz = ds.data_frame_kurz.iloc[:-10]
                    ds.data_frame_long = ds.data_frame_long.iloc[:-10]

                    cf_ds_kurz = ds.data_frame_kurz.iloc[-10:].copy()
                    cf_ds_long = ds.data_frame_long.iloc[-10:].copy()

                    standard_deviations = list(ds.data_frame_kurz.std())
                    observable_data_dict = ds.data_frame_kurz.copy().T.to_dict()

                    layerStructure = []
                    for layer in model_keras.layers:
                        layerStructure.append(layer.get_weights()[0].shape[0])

                    model_skl = MLPClassifier(layerStructure[1:], max_iter=1)

                    model_skl.fit(ds.data_frame_kurz.drop(columns=['y']).to_numpy(), ds.data_frame_kurz[['y']].to_numpy())

                    for layer_idx, layer in enumerate(model_keras.layers):
                        model_skl.coefs_[layer_idx] = layer.get_weights()[0]
                        model_skl.intercepts_[layer_idx] = layer.get_weights()[1]

                    factual_sample = cf_ds_kurz.iloc[idx_cf].to_dict()
                    factual_sample['y'] = True if factual_sample['y'] else False

                    print(c, dsName)
                    start_time = time.time()
                    cf_out = generateSATExplanations.genExp(
                        'test',
                        model_skl,
                        ds,
                        factual_sample,
                        'zero_norm',
                        'mace',
                        1e-5
                    )

                    if cf_out is not None:
                        cfs_SEDC.append(pd.DataFrame([cf_out['cfe_sample']]).drop(columns=['y']).to_numpy()[0][1:])

                        timeRunMACE = [idx_cf, int(c), dsName, time.time() - start_time]
                        pd.DataFrame(timeRunMACE).T.to_csv('../cfoutput/TIME_MACE.csv', mode='a', header=False,
                                                           index=False)
                        print('Found!', dsName, idx_cf)
                    else:
                        cfs_SEDC.append([np.NaN] * (cf_ds_kurz.shape[1]-1))
                        print('Not found', dsName, idx_cf)

                    if cat_feats:
                        del converter, dict_conversion,
                    del bin_feats, df_input, ds, cf_ds_kurz, cf_ds_long, standard_deviations, observable_data_dict, \
                        model_skl, factual_sample, cf_out
                    gc.collect()

                pd.DataFrame(cfs_SEDC).to_csv(f'../cfoutput/{str(int(c))}_{dsName}_MACE.csv', index=False)
        except FileNotFoundError:
            print(f'{str(int(c))}_{dsName} has no data')