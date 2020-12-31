import random as python_random

import time
import sys
sys.path.append('../frameworks/LORE/')

import tensorflow as tf
from tensorflow.keras.models import load_model

import pandas as pd
import numpy as np

from frameworks.LORE import test_lore
from benchmark.cfg import CFmodel
from benchmark.cfg import OHConverter

from constants.var_types import VAR_TYPES


# Set seeds
np.random.seed(42)
python_random.seed(42)
tf.random.set_seed(42)


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

                cfs_LORE = []
                for idx_cf in range(cf.shape[0]):
                    # Start to measure time to get CF
                    print(c, dsName)
                    start_time = time.time()

                    converterLORE = OHConverter.Converter(df,
                                                          cat_feats,
                                                          list(df_oh.columns if len(df_oh) > 0 else df.columns))

                    cfLORE = test_lore.main(df_train,
                                            cat_feats,
                                            'output',
                                            list(df_oh.columns if len(df_oh) > 0 else df.columns),
                                            model,
                                            cf.iloc[idx_cf]
                                            )

                    # Measure time to get CF
                    timeRunLORE = [idx_cf, int(c), dsName, time.time() - start_time]
                    if len(cfLORE) > 0:
                        pd.DataFrame(timeRunLORE).T.to_csv('../cfoutput/TIME_LORE.csv', mode='a', header=False, index=False)

                        cfs_LORE.append(cfLORE.to_numpy()[0].tolist())
                    else:
                        data_size = df_oh.shape[1] - 1 if len(df_oh) > 0 else df.shape[1] - 1
                        cfs_LORE.append([np.NaN] * data_size)

                try:
                    pd.DataFrame(cfs_LORE).to_csv(f'../cfoutput/{str(int(c))}_{dsName}_LORE.csv', index=False)
                except ValueError:
                    print('No results')
        except FileNotFoundError:
            print(f'{str(int(c))}_{dsName} has no data')
