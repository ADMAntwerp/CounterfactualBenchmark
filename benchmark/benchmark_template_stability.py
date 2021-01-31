import random as python_random
import operator
import os

verbose = os.getenv('VERBOSE')

if verbose == '0':
    import logging
    logging.getLogger("tensorflow").setLevel(logging.ERROR)

import sys
if verbose == '0':
    sys.stdout = open(os.devnull, 'w')
sys.path.append('../')

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

import pandas as pd
import numpy as np

from benchmark.cfg import OHConverter

from constants.var_types import VAR_TYPES

# Set seeds
np.random.seed(42)
python_random.seed(42)
try:
    tf.random.set_random_seed(42)
except AttributeError:
    tf.random.set_seed(42)
tf.compat.v1.disable_v2_behavior()  # disable TF2 behaviour as alibi code still relies on TF1 constructs

print('TF version: ', tf.__version__)
print('Eager execution enabled: ', tf.executing_eagerly())  # False

# Disable GPU
# Disable all GPUS
try:
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except AttributeError:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def _nn_ohe(input_shape, hidden_layers_ws, output_number):
    x_in = Input(shape=(input_shape,))
    x = Dense(hidden_layers_ws, activation='relu')(x_in)
    x_out = Dense(2, activation='softmax')(x)
    if output_number == 1:
        x_bin = Dense(1, activation='linear')(x_out)
        nn = Model(inputs=x_in, outputs=x_bin)
    if output_number == 2:
        nn = Model(inputs=x_in, outputs=x_out)

    nn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return nn


VAR_TYPES_IDX = [*range(len(VAR_TYPES))]


def run_experiment(framework_name, framework_tester, output_number=1, ds_id_test=VAR_TYPES_IDX):

    # Get datasets to test from the ds_id_test list
    datasets_to_test = operator.itemgetter(*ds_id_test)(list(VAR_TYPES.keys()))
    if len(ds_id_test) == 1:
        datasets_to_test = [datasets_to_test]

    for dsName in datasets_to_test:

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

            # Load factual data
            df_factual = pd.read_csv(f'../experiments_data/{dsName}_CFDATASET_{c}.csv').drop(columns=['Unnamed: 0'])

            try:
                # Load the last generated counterfactuals
                df_last_cf = pd.read_csv(f"../cfoutput/{str(int(c))}_{dsName}_{framework_name}.csv")
                # Get only the lines that generated cf
                df_last_cf = df_last_cf.dropna()
                # Get 2 index
                idx_to_test = df_last_cf.sample(10, random_state=42).index.to_list()

                # Save verified indexes
                with open(f'../cfoutput/STABILITY_TEST_IDX_{str(int(c))}_{dsName}_{framework_name}.txt', 'w') as f:
                    f.write(str(idx_to_test))

                # Get the data
                df_factual = df_factual.iloc[idx_to_test]

                # Load train data
                df_train = pd.read_csv(f'../experiments_data/{dsName}_TRAINDATASET.csv').drop(columns=['Unnamed: 0'])
                # Load test data
                df_test = pd.read_csv(f'../experiments_data/{dsName}_TESTDATASET.csv').drop(columns=['Unnamed: 0'])
                # Load OH if existent
                if cat_feats:
                    # Load train data
                    df_oh_train = pd.read_csv(f'../experiments_data/{dsName}_TRAINOHDATASET.csv').drop(
                        columns=['Unnamed: 0'])
                    # Load test data
                    df_oh_test = pd.read_csv(f'../experiments_data/{dsName}_TESTOHDATASET.csv').drop(columns=['Unnamed: 0'])
                else:
                    df_oh_train = df_train.copy()
                    df_oh_test = df_test.copy()

                if c == 1:
                    # Revert
                    df_factual['output'] = df_factual['output'].map({1: 0, 0: 1})

                    df_train['output'] = df_train['output'].map({1: 0, 0: 1})
                    df_test['output'] = df_test['output'].map({1: 0, 0: 1})

                    df_oh_train['output'] = df_oh_train['output'].map({1: 0, 0: 1})
                    df_oh_test['output'] = df_oh_test['output'].map({1: 0, 0: 1})

                cfs_framework = []
                for idx_cf in range(df_factual.shape[0]):

                    with tf.compat.v1.Session() as session:

                        # Load model
                        model_keras = load_model(f'../models/{dsName}.h5', compile=False)

                        # Get architecture info and pass to our NN architecture
                        input_shape = model_keras.get_weights()[0].shape[0]
                        hidden_layers_ws = model_keras.get_weights()[0].shape[1]

                        adapted_nn = _nn_ohe(input_shape, hidden_layers_ws, output_number)
                        adapted_nn.build(input_shape=input_shape)

                        adapted_nn.layers[1].set_weights(model_keras.layers[0].get_weights())
                        adapted_nn.layers[2].set_weights(model_keras.layers[1].get_weights())

                        if c == 1:
                            # If the tested class is the minority, revert output order
                            second_layer_w = []
                            for w_row in model_keras.layers[1].get_weights()[0]:
                                second_layer_w.append(np.flip(w_row).tolist())

                            second_layer_b = np.flip(model_keras.layers[1].get_weights()[1])

                            second_layer = [np.array(second_layer_w, dtype=np.float32),
                                            np.array(second_layer_b, dtype=np.float32)]
                            adapted_nn.layers[2].set_weights(second_layer)

                        # The original model has 2 outputs, therefore, the weights must be modified to output only one
                        if output_number == 1:
                            adapted_nn.layers[3].set_weights(
                                [np.array([[0.0], [1.0]], dtype=np.float32), np.array([0.0], dtype=np.float32)])

                        # If there are categorical features, create a CF OH entry
                        if cat_feats:
                            converter = OHConverter.Converter(df, cat_feats, list(df_oh.columns))
                        else:
                            converter = lambda x: x

                        factual = df_factual.drop(columns=['output']).iloc[idx_cf].to_list()
                        factual_oh = converter.convert_to_oh(df_factual.drop(columns=['output']).iloc[idx_cf].to_list()) if cat_feats else factual

                        print(f'CF GENERATION: DATA - {dsName} / C - {c} /ROW - {idx_cf}')
                        cf_out, cf_time = framework_tester(df_train, df_oh_train, df_test, df_oh_test, num_feats, cat_feats,
                                                           converter, adapted_nn, df_factual, factual, factual_oh, session)

                        # Verify if counterfactual (cf) is a list, the output MUST a list
                        try:
                            assert type(cf_out) == list
                        except AssertionError:
                            raise AssertionError('The counterfactual output must be a list')
                        # Verify if counterfactual (cf) is one-hot encoded or not, output must be always one-hot encoded
                        try:
                            assert len(cf_out) in [len(factual), len(factual_oh), 0]
                        except AssertionError:
                            raise AssertionError(f'The counterfactual output must be the same size as the factual '
                                                 f'({len(factual)}) or one-hot encoded factual ({len(factual_oh)}) or zero')

                        # Measure time to get CF only if we get a cf
                        if len(cf_out) > 0:
                            timeRunFramework = [idx_cf, int(c), dsName, cf_time]
                            pd.DataFrame(timeRunFramework).T.to_csv(f'../cfoutput/STABILITY_TIME_{framework_name}.csv', mode='a',
                                                                    header=False, index=False)

                        # Append result to result array and verify if it's a counterfactual
                        if len(cf_out) > 0:

                            # Saved result must always be one-hot encoded
                            if len(cf_out) != len(factual_oh):
                                cf_out_oh = converter.convert_to_oh(factual)
                            else:
                                cf_out_oh = cf_out

                            if cat_feats:
                                factual_class = adapted_nn.predict(np.array([factual_oh]))[0]
                                cf_class = adapted_nn.predict(np.array([cf_out_oh]))[0]
                            else:
                                factual_class = adapted_nn.predict(np.array([factual_oh]))[0]
                                cf_class = adapted_nn.predict(np.array([cf_out_oh]))[0]

                            if output_number == 2:
                                factual_class = factual_class[1]
                                cf_class = cf_class[1]
                            else:
                                factual_class = factual_class[0]
                                cf_class = cf_class[0]

                            # Check if the factual class is indeed 0, if not, it's a internal problem that must be reported
                            try:
                                assert factual_class < 0.5
                            except AssertionError:
                                raise AssertionError('Internal error, original class is wrong. Please, report this error.')

                            if cf_class >= 0.5:
                                print(f'Counterfactual found!\nFactual class:{factual_class}\nCF class:{cf_class}\n')
                            else:
                                print(f'Failed counterfactual!\nFactual class:{factual_class}\nCF class:{cf_class}\n')

                            cfs_framework.append(cf_out_oh)
                        else:
                            print(f'No returned counterfactual candidate!\n')
                            cfs_framework.append([np.NaN] * (len(factual_oh)))

                    tf.keras.backend.clear_session()

                pd.DataFrame(cfs_framework).to_csv(f'../cfoutput/STABILITY_{str(int(c))}_{dsName}_{framework_name}.csv', index=False)
            except FileNotFoundError:
                with open(f'./STABILITY_LOG_{framework_name}.txt', 'a') as f:
                    f.write(str(f'NO RESULTS FILE FOUND FOR DATASET {dsName} AND CLASS {str(int(c))} \n'))
