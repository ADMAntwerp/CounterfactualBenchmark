import random as python_random

import time
import sys
sys.path.append('../frameworks/explanations/')

import tensorflow as tf
from tensorflow.keras.models import load_model

import pandas as pd
import numpy as np

import explainer
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

                cfs_SEDC = []
                for idx_cf in range(cf.shape[0]):

                    decision_boundary = 0.5

                    if cat_feats:
                        default_values = df_oh_train.drop(columns=['output']).mean(axis=0)

                        converter = OHConverter.Converter(df, cat_feats, list(df_oh.columns))
                        cf_oh = converter.convert_to_oh(cf.drop(columns=['output']).iloc[idx_cf].to_list())
                    else:
                        default_values = df_train.drop(columns=['output']).mean(axis=0)

                    def scoring_function(X):
                        output = []
                        for res in model_keras.predict_proba(X):
                            output.append([1-res[0]])

                        return np.array(output)


                    explain = explainer.Explainer(scoring_function, default_values)

                    # Start to measure time to get CF
                    start_time = time.time()

                    if cat_feats:
                        explanations = explain.explain(np.array([cf_oh]), decision_boundary)
                    else:
                        explanations = explain.explain(np.array([cf.drop(columns=['output']).iloc[idx_cf].to_numpy()]),
                                                       decision_boundary)

                    # Measure time to get CF
                    timeRunSEDC = [idx_cf, int(c), dsName, time.time() - start_time]

                    if explanations[0]:

                        pd.DataFrame(timeRunSEDC).T.to_csv('../cfoutput/TIME_SEDC.csv', mode='a', header=False,
                                                             index=False)
                        if cat_feats:
                            cf_point = np.array(cf_oh.copy())
                        else:
                            cf_point = np.array([cf.drop(columns=['output']).iloc[idx_cf].to_numpy()][0]).copy()

                        for feat_change in explanations[0]:
                            cf_point[feat_change] = default_values[feat_change]
                        cfs_SEDC.append(cf_point)
                    else:
                        if cat_feats:
                            cfs_SEDC.append([np.NaN] * (df_oh.shape[1] - 1))
                        else:
                            cfs_SEDC.append([np.NaN] * (df.shape[1] - 1))

                pd.DataFrame(cfs_SEDC).to_csv(f'../cfoutput/{str(int(c))}_{dsName}_SEDC.csv', index=False)
        except FileNotFoundError:
            print(f'{str(int(c))}_{dsName} has no data')