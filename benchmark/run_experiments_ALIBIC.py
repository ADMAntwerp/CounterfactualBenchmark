import random as python_random

import time
import sys
sys.path.append('../frameworks/alibi/')

import tensorflow as tf
from tensorflow.keras.models import load_model

import pandas as pd
import numpy as np

from alibi.explainers.cfproto import CounterFactualProto
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

            cfs_DiCE = []
            for idx_cf in range(cf.shape[0]):

                # If there are categorical features, create a CF OH entry
                dict_oh = {}
                if cat_feats:
                    converter = OHConverter.Converter(df, cat_feats, list(df_oh.columns))
                    cf_oh = converter.convert_to_oh(cf.drop(columns=['output']).iloc[idx_cf].to_list())

                    # If there are categorical features, prepare the one-hot encoding dictionary
                    prefix_oh_list = [int(float(l.split('_')[0])) for l in list(df_oh.drop(columns='output').columns)]
                    for cat_f in list(set(cat_feats)-set(converter.binary_cats)):
                        init_idx = prefix_oh_list.index(int(float(cat_f)))
                        len_oh = len(df[cat_f].unique())
                        dict_oh[init_idx] = len_oh

                model_keras.add(tf.keras.layers.Dense(2, activation='softmax'))

                # Additional model layer to input a two inputs instead one binary
                model_keras.layers[5].set_weights([np.array([[-1,1]]),np.array([1,0])])

                selected_df_train = df_train if not cat_feats else df_oh_train

                # Define feature range
                minFRang = [df_train.drop(columns=['output']).min(axis=0).to_list()]
                maxFRang = [df_train.drop(columns=['output']).max(axis=0).to_list()]
                for idx, cat_f in enumerate(cat_feats):
                    minFRang[0][idx] = 0
                    maxFRang[0][idx] = 1
                minFRang = np.array(minFRang)
                maxFRang = np.array(maxFRang)


                alibi_cfs = CounterFactualProto(model_keras,
                                                (1, df_oh.shape[1] - 1 if cat_feats else df.shape[1] - 1),
                                                use_kdtree=True,
                                                feature_range=(minFRang, maxFRang),
                                                cat_vars=dict_oh if dict_oh else None,
                                                ohe=True if cat_feats and dict_oh else False)

                try:
                    alibi_cfs.fit(selected_df_train.drop(columns=['output']).to_numpy())

                    # Start to measure time to get CF
                    start_time = time.time()

                    if not cat_feats:
                        exp_cfs = alibi_cfs.explain(
                            np.array([cf.drop(columns=['output']).iloc[idx_cf].to_numpy().tolist()]))
                    else:
                        exp_cfs = alibi_cfs.explain(np.array([cf_oh]))

                    # Measure time to get CF
                    timeRunALIBIC = [idx_cf, int(c), dsName, time.time() - start_time]

                    if exp_cfs.all:
                        pd.DataFrame(timeRunALIBIC).T.to_csv('../cfoutput/TIME_ALIBIC.csv', mode='a', header=False,
                                                             index=False)
                        cfs_DiCE.append(exp_cfs.data['cf']['X'][0])
                    else:
                        if cat_feats:
                            cfs_DiCE.append([np.NaN] * df_oh.shape[1])
                        else:
                            cfs_DiCE.append([np.NaN] * df.shape[1])

                except ValueError:
                    print('CF NOT OUTPUT DUE TO INTERNAL ERRORS')
                    cfs_DiCE.append([np.NaN] * df.shape[1])

            pd.DataFrame(cfs_DiCE).to_csv(f'../cfoutput/{str(int(c))}_{dsName}_ALIBIC.csv', index=False)
