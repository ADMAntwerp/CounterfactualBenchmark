import random as python_random

import time
import sys
sys.path.append('../frameworks/DiCE/')

import tensorflow as tf
from tensorflow.keras.models import load_model

import pandas as pd
import numpy as np

from benchmark.wrappers import DiCE
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

            cfs_DiCE = []
            for idx_cf in range(cf.shape[0]):

                # Start to measure time to get CF
                start_time = time.time()

                cfs = DiCE.get_cf(df_train, num_feats, cat_feats, model_keras, cf.drop(columns=['output']).iloc[idx_cf])

                # Measure time to get CF
                timeRunDiCE = [idx_cf, int(c), dsName, time.time() - start_time]
                pd.DataFrame(timeRunDiCE).T.to_csv('../cfoutput/TIME_LORE.csv', mode='a', header=False, index=False)

                cf_out = [idx_cf] + list(cfs.final_cfs_sparse[0][0])
                if cat_feats:
                    converter = OHConverter.Converter(df, cat_feats, list(df_oh.columns))
                    cf_out = [idx_cf] + converter.convert(cfs.final_cfs_sparse[0])[0]

                cfs_DiCE.append(cf_out)

            pd.DataFrame(cfs_DiCE).to_csv(f'../cfoutput/{str(int(c))}_{dsName}_DiCE.csv', index=False)
