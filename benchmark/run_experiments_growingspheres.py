import random as python_random

import time
import sys
sys.path.append('../frameworks/highgarden/highgarden/exploration')

import tensorflow as tf
from tensorflow.keras.models import load_model

import pandas as pd
import numpy as np

import uniform_growing_spheres
import path_to_ennemies
import uniform_growing_spheres_featsel
import path_to_ennemies_featsel

from benchmark.cfg import CFmodel
from benchmark.cfg import OHConverter

from constants.var_types import VAR_TYPES


# Set seeds
np.random.seed(42)
python_random.seed(42)
tf.random.set_random_seed(42)
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

            cfs_GrowingS1 = []
            cfs_GrowingS2 = []
            cfs_GrowingS3 = []
            cfs_GrowingS4 = []

            for idx_cf in range(cf.shape[0]):

                def pred(obs):
                    if len(obs.shape) > 1 and obs.shape[1] > 1:
                        return np.array([x[0] for x in model_keras.predict_proba(obs)])
                    else:
                        return model_keras.predict_proba(obs.reshape(1, -1))[0][0]

                def interpretability_metrics(X, prediction_function, interpretability_method, obs_to_interprete,
                                             **kwargs):
                    nearest_ennemy = interpretability_method(X, prediction_function, obs_to_interprete, **kwargs)
                    return nearest_ennemy

                if cat_feats:
                    df_train_data = df_oh_train.drop(columns=['output']).to_numpy()
                    converter = OHConverter.Converter(df, cat_feats, list(df_oh.columns))
                    cf_data = np.array(converter.convert_to_oh(cf.drop(columns=['output']).iloc[idx_cf].to_list()))
                else:
                    df_train_data = df_train.drop(columns=['output']).to_numpy()
                    cf_data = np.array(cf.drop(columns=['output']).iloc[idx_cf].to_numpy())

                # Start to measure time to get CF
                start_time1 = time.time()
                cf_out1 = interpretability_metrics(df_train_data,
                                                   pred,
                                                   uniform_growing_spheres.main,
                                                   cf_data)

                if (len(cf_out1) > 0):
                    cfs_GrowingS1.append(cf_out1)

                    timeRunGrowingS1 = [idx_cf, int(c), dsName, time.time() - start_time1]
                    pd.DataFrame(timeRunGrowingS1).T.to_csv('../cfoutput/TIME_GROWINGS1.csv', mode='a', header=False,
                                                            index=False)
                else:
                    cfs_GrowingS1.append([np.NaN] * cf_data.shape[0])



                # Start to measure time to get CF
                start_time2 = time.time()
                cf_out2 = interpretability_metrics(df_train_data,
                                                   pred,
                                                   path_to_ennemies.main,
                                                   cf_data)

                if (len(cf_out2) > 0):
                    cfs_GrowingS2.append(cf_out2)

                    timeRunGrowingS2 = [idx_cf, int(c), dsName, time.time() - start_time2]
                    pd.DataFrame(timeRunGrowingS2).T.to_csv('../cfoutput/TIME_GROWINGS2.csv', mode='a', header=False,
                                                            index=False)
                else:
                    cfs_GrowingS2.append([np.NaN] * cf_data.shape[0])

                # Start to measure time to get CF
                start_time3 = time.time()
                cf_out3 = interpretability_metrics(df_train_data,
                                                   pred,
                                                   uniform_growing_spheres_featsel.main,
                                                   cf_data)

                if (len(cf_out3) > 0):
                    cfs_GrowingS3.append(cf_out3)

                    timeRunGrowingS3 = [idx_cf, int(c), dsName, time.time() - start_time3]
                    pd.DataFrame(timeRunGrowingS3).T.to_csv('../cfoutput/TIME_GROWINGS3.csv', mode='a', header=False,
                                                            index=False)
                else:
                    cfs_GrowingS3.append([np.NaN] * cf_data.shape[0])

                # Start to measure time to get CF
                start_time4 = time.time()
                cf_out4 = interpretability_metrics(df_train_data,
                                                   pred,
                                                   path_to_ennemies_featsel.main,
                                                   cf_data)

                if (len(cf_out4) > 0):
                    cfs_GrowingS4.append(cf_out4)

                    timeRunGrowingS4 = [idx_cf, int(c), dsName, time.time() - start_time4]
                    pd.DataFrame(timeRunGrowingS4).T.to_csv('../cfoutput/TIME_GROWINGS4.csv', mode='a', header=False,
                                                            index=False)
                else:
                    cfs_GrowingS4.append([np.NaN] * cf_data.shape[0])

            pd.DataFrame(cfs_GrowingS1).to_csv(f'../cfoutput/{str(int(c))}_{dsName}_GROWINGS1.csv', index=False)
            pd.DataFrame(cfs_GrowingS2).to_csv(f'../cfoutput/{str(int(c))}_{dsName}_GROWINGS2.csv', index=False)
            pd.DataFrame(cfs_GrowingS3).to_csv(f'../cfoutput/{str(int(c))}_{dsName}_GROWINGS3.csv', index=False)
            pd.DataFrame(cfs_GrowingS4).to_csv(f'../cfoutput/{str(int(c))}_{dsName}_GROWINGS4.csv', index=False)