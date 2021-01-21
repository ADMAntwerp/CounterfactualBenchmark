import random as python_random

import time
import sys
sys.path.append('../frameworks/alibi/')
sys.path.append('../')

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

import pandas as pd
import numpy as np

from alibi.explainers.cfproto import CounterFactualProto
from benchmark.cfg import CFmodel
from benchmark.cfg import OHConverter

# from constants.var_types import VAR_TYPES
from constants.var_types import VAR_TYPES




# Set seeds
np.random.seed(42)
python_random.seed(42)
tf.random.set_seed(42)
tf.compat.v1.disable_v2_behavior() # disable TF2 behaviour as alibi code still relies on TF1 constructs

print('TF version: ', tf.__version__)
print('Eager execution enabled: ', tf.executing_eagerly()) # False

# Disable GPU
# Disable all GPUS
tf.config.set_visible_devices([], 'GPU')
visible_devices = tf.config.get_visible_devices()
for device in visible_devices:
    assert device.device_type != 'GPU'

def nn_ohe(input_shape, hidden_layers_ws):

    x_in = Input(shape=(input_shape,))
    x = Dense(hidden_layers_ws, activation='relu')(x_in)
    x_out = Dense(2, activation='softmax')(x)

    nn = Model(inputs=x_in, outputs=x_out)
    nn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return nn


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
    for c in [0, 1]:


        cf = pd.read_csv(f'../experiments_data/{dsName}_CFDATASET_{c}.csv').drop(columns=['Unnamed: 0'])

        # If there are CF to be analyzed
        if cf.shape[0] > 0:

            # Load train data
            df_train = pd.read_csv(f'../experiments_data/{dsName}_TRAINDATASET.csv').drop(columns=['Unnamed: 0'])
            # Load test data
            df_test = pd.read_csv(f'../experiments_data/{dsName}_TESTDATASET.csv').drop(columns=['Unnamed: 0'])
            # Load OH if existent
            if cat_feats:
                # Load train data
                df_oh_train = pd.read_csv(f'../experiments_data/{dsName}_TRAINOHDATASET.csv').drop(columns=['Unnamed: 0'])
                # Load test data
                df_oh_test = pd.read_csv(f'../experiments_data/{dsName}_TESTOHDATASET.csv').drop(columns=['Unnamed: 0'])

            if c == 1:
                # Revert
                cf['output'] = cf['output'].map({1:0, 0:1})
                df_train['output'] = df_train['output'].map({1:0, 0:1})
                df_test['output'] = df_test['output'].map({1:0, 0:1})

                if cat_feats:
                    df_oh_train['output'] = df_oh_train['output'].map({1: 0, 0: 1})
                    df_oh_test['output'] = df_oh_test['output'].map({1: 0, 0: 1})

            cfs_DiCE = []
            for idx_cf in range(cf.shape[0]):

                # Load model
                model_keras = load_model(f'../models/{dsName}.h5', compile=False)
                model = CFmodel.CreateModel(dsName, str(int(c)))

                input_shape = model_keras.get_weights()[0].shape[0]
                hidden_layers_ws = model_keras.get_weights()[0].shape[1]

                adapted_nn = nn_ohe(input_shape, hidden_layers_ws)
                adapted_nn.build(input_shape=input_shape)

                adapted_nn.layers[1].set_weights(model_keras.layers[0].get_weights())
                adapted_nn.layers[2].set_weights(model_keras.layers[1].get_weights())

                if c == 1:
                    # Revert
                    second_layer_w = []
                    for w_row in model_keras.layers[1].get_weights()[0]:
                        second_layer_w.append(np.flip(w_row).tolist())

                    second_layer_b = np.flip(model_keras.layers[1].get_weights()[1])

                    second_layer = [np.array(second_layer_w, dtype=np.float32),
                                    np.array(second_layer_b, dtype=np.float32)]

                print(idx_cf)

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

                selected_df_train = df_train if not cat_feats else df_oh_train

                # Define feature range
                minFRang = [df_train.drop(columns=['output']).min(axis=0).to_list()]
                maxFRang = [df_train.drop(columns=['output']).max(axis=0).to_list()]
                for idx, cat_f in enumerate(cat_feats):
                    minFRang[0][idx] = 0
                    maxFRang[0][idx] = 1
                minFRang = np.array(minFRang)
                maxFRang = np.array(maxFRang)


                alibi_cfs = CounterFactualProto(lambda x : adapted_nn.predict(x),
                                                (1, df_oh.shape[1] - 1 if cat_feats else df.shape[1] - 1),
                                                use_kdtree=True,
                                                feature_range=(minFRang, maxFRang),
                                                cat_vars=dict_oh if dict_oh else None,
                                                ohe=True if cat_feats and dict_oh else False)

                alibi_cfs.fit(selected_df_train.drop(columns=['output']).to_numpy())

                print(c, dsName)
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
                    if len(cat_feats):
                        print(
                            f"ORIGINAL: {adapted_nn.predict(np.array([converter.convert_to_oh(cf.drop(columns=['output']).iloc[idx_cf].to_numpy().tolist())]))}"
                            f"CF: {adapted_nn.predict(np.array([exp_cfs.data['cf']['X'][0]]))}")
                    else:
                        print(
                            f"ORIGINAL: {adapted_nn.predict(np.array([cf.drop(columns=['output']).iloc[idx_cf].to_numpy().tolist()]))}"
                            f"CF: {adapted_nn.predict(np.array([exp_cfs.data['cf']['X'][0]]))}")

                    pd.DataFrame(timeRunALIBIC).T.to_csv('../cfoutput/TIME_ALIBICNOGRAD.csv', mode='a', header=False,
                                                         index=False)
                    cfs_DiCE.append(exp_cfs.data['cf']['X'][0])
                else:
                    print('NO CF')
                    if cat_feats:
                        cfs_DiCE.append([np.NaN] * (df_oh.shape[1] - 1))
                    else:
                        cfs_DiCE.append([np.NaN] * (df.shape[1] - 1))

                tf.keras.backend.clear_session()

            pd.DataFrame(cfs_DiCE).to_csv(f'../cfoutput/{str(int(c))}_{dsName}_ALIBICNOGRAD.csv', index=False)


