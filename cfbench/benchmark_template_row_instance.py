import random as python_random
import logging
import os

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from keras import backend

import pandas as pd
import numpy as np

from cfbench.cfg import OHConverter

from cfbench.dataset_data.constants.var_types import VAR_TYPES

CURRENT_PATH = '/'.join(os.path.abspath(__file__).split('/')[:-1])

logging.basicConfig(level=logging.INFO)

# Set seeds
np.random.seed(42)
python_random.seed(42)
try:
    tf.random.set_random_seed(42)
except AttributeError:
    tf.random.set_seed(42)
tf.compat.v1.disable_v2_behavior()  # disable TF2 behaviour as alibi code still relies on TF1 constructs


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


class BenchmarkGenerator:

    def __init__(
            self,
            framework_name,
            output_number,
            ds_id_test,
            disable_gpu):
        self.framework_name = framework_name
        self.output_number = output_number
        self.ds_id_tes = ds_id_test

        self.ds_idx = 0
        self.current_dsName = None
        self.factual_class = 0
        self.factual_idx = 0
        self.total_factuals = None

        if disable_gpu:
            # Disable GPU
            # Disable all GPUS
            try:
                tf.config.set_visible_devices([], 'GPU')
                visible_devices = tf.config.get_visible_devices()
                for device in visible_devices:
                    assert device.device_type != 'GPU'
            except AttributeError:
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    def __iter__(self):
        return self

    def __next__(self):

        if self.factual_idx + 1 == self.total_factuals and self.total_factuals is not None:
            # Reset factual index
            self.factual_idx = 0

            # First, we change the class, but if the class is already one, we reset and go to the next dataset
            if self.factual_class == 0:
                self.factual_class = 1
            else:
                # Go back to first class
                self.factual_class = 0
                # Go to next dataset IDX
                self.ds_idx += 1
                # Verify if this a valid dataset index
                if self.ds_idx >= len(self.ds_id_tes):
                    raise StopIteration

        dsName = list(VAR_TYPES.keys())[self.ds_id_tes[self.ds_idx]]

        if dsName == self.current_dsName:
            # Go to next row
            self.factual_idx += 1
        else:
            # Assign new dataset name
            self.current_dsName = dsName

            # Load feature type specifications
            self.cat_feats = VAR_TYPES[dsName]['categorical']
            self.num_feats = VAR_TYPES[dsName]['numerical']

            # Load Dataset
            if self.cat_feats and self.num_feats:
                df = pd.read_csv(f'{CURRENT_PATH}/dataset_data/data/NORM_{dsName}.csv')
                df_oh = pd.read_csv(f'{CURRENT_PATH}/dataset_data/data/OH_NORM_{dsName}.csv')
            elif self.cat_feats:
                df = pd.read_csv(f'{CURRENT_PATH}/dataset_data/data/{dsName}.csv')
                df_oh = pd.read_csv(f'{CURRENT_PATH}/dataset_data/data/OH_{dsName}.csv')
            else:
                df = pd.read_csv(f'{CURRENT_PATH}/dataset_data/data/NORM_{dsName}.csv')
                df_oh = []

            df_y_original = df['output'].copy()
            df_oh_y_original = df['output'].copy()

            # Load factual data
            self.df_factual = pd.read_csv(
                f'{CURRENT_PATH}/dataset_data/experiments_data/{dsName}_CFDATASET_{self.factual_class}.csv').drop(
                columns=['Unnamed: 0'])
            self.total_factuals = self.df_factual.shape[0]

            # Load train data
            self.df_train = pd.read_csv(
                f'{CURRENT_PATH}/dataset_data/experiments_data/{dsName}_TRAINDATASET.csv').drop(
                columns=['Unnamed: 0'])
            # Load test data
            self.df_test = pd.read_csv(
                f'{CURRENT_PATH}/dataset_data/experiments_data/{dsName}_TESTDATASET.csv').drop(
                columns=['Unnamed: 0'])
            # Load OH if existent
            if self.cat_feats:
                # Load train data
                self.df_oh_train = pd.read_csv(
                    f'{CURRENT_PATH}/dataset_data/experiments_data/{dsName}_TRAINOHDATASET.csv').drop(
                    columns=['Unnamed: 0'])
                # Load test data
                self.df_oh_test = pd.read_csv(
                    f'{CURRENT_PATH}/dataset_data/experiments_data/{dsName}_TESTOHDATASET.csv').drop(
                    columns=['Unnamed: 0'])
            else:
                self.df_oh_train = self.df_train.copy()
                self.df_oh_test = self.df_test.copy()

            if self.factual_class == 1:
                # Revert
                self.df_factual['output'] = self.df_factual['output'].map({1: 0, 0: 1})

                self.df_train['output'] = self.df_train['output'].map({1: 0, 0: 1})
                self.df_test['output'] = self.df_test['output'].map({1: 0, 0: 1})

                self.df_oh_train['output'] = self.df_oh_train['output'].map({1: 0, 0: 1})
                self.df_oh_test['output'] = self.df_oh_test['output'].map({1: 0, 0: 1})

            # Load model
            model_keras = load_model(f'{CURRENT_PATH}/model_data/models/{dsName}.h5', compile=False)

            # Get session
            self.tf_session = backend.get_session()

            # Get architecture info and pass to our NN architecture
            input_shape = model_keras.get_weights()[0].shape[0]
            hidden_layers_ws = model_keras.get_weights()[0].shape[1]

            self.adapted_nn = _nn_ohe(input_shape, hidden_layers_ws, self.output_number)
            self.adapted_nn.build(input_shape=input_shape)

            self.adapted_nn.layers[1].set_weights(model_keras.layers[0].get_weights())
            self.adapted_nn.layers[2].set_weights(model_keras.layers[1].get_weights())

            if self.factual_class == 1:
                # If the tested class is the minority, revert output order
                second_layer_w = []
                for w_row in model_keras.layers[1].get_weights()[0]:
                    second_layer_w.append(np.flip(w_row).tolist())

                second_layer_b = np.flip(model_keras.layers[1].get_weights()[1])

                second_layer = [np.array(second_layer_w, dtype=np.float32),
                                np.array(second_layer_b, dtype=np.float32)]
                self.adapted_nn.layers[2].set_weights(second_layer)

            # The original model has 2 outputs, therefore, the weights must be modified to output only one
            if self.output_number == 1:
                self.adapted_nn.layers[3].set_weights(
                    [np.array([[0.0], [1.0]], dtype=np.float32), np.array([0.0], dtype=np.float32)])

            # If there are categorical features, create a CF OH entry
            if self.cat_feats:
                self.converter = OHConverter.Converter(df, self.cat_feats, list(df_oh.columns))
            else:
                self.converter = lambda x: x

        self.factual = self.df_factual.drop(columns=['output']).iloc[self.factual_idx].to_list()
        self.factual_oh = self.converter.convert_to_oh(
            self.df_factual.drop(columns=['output']).iloc[self.factual_idx].to_list()) \
            if self.cat_feats else self.factual

        return {
            'df_train': self.df_train,
            'df_oh_train': self.df_oh_train,
            'df_test': self.df_test,
            'df_oh_test': self.df_oh_test,
            'num_feats': self.num_feats,
            'cat_feats': self.cat_feats,
            'oh_converter': self.converter,
            'model': self.adapted_nn,
            'df_factual': self.df_factual,
            'factual': self.factual,
            'factual_oh': self.factual_oh,
            'tf_session': self.tf_session,

            'cf_evaluator': self.cf_evaluator,

            'factual_idx': self.factual_idx,
            'factual_class': self.factual_class,
            'dsname': self.current_dsName
        }

    def cf_evaluator(self, cf_out: list, verbose: bool = False):
        # Verify if counterfactual (cf) is a list, the output MUST a list
        try:
            assert type(cf_out) == list
        except AssertionError:
            raise AssertionError('The counterfactual output must be a list')
        # Verify if counterfactual (cf) is one-hot encoded or not, output must be always one-hot encoded
        try:
            assert len(cf_out) in [len(self.factual), len(self.factual_oh), 0]
        except AssertionError:
            raise AssertionError(f'The counterfactual output must be the same size as the factual '
                                 f'({len(self.factual)}) or one-hot encoded factual ({len(self.factual_oh)}) or zero')

        # Append result to result array and verify if it's a counterfactual
        if len(cf_out) > 0:

            # Saved result must always be one-hot encoded
            if len(cf_out) != len(self.factual_oh):
                cf_out_oh = self.converter.convert_to_oh(self.factual)
            else:
                cf_out_oh = cf_out

            if self.cat_feats:
                factual_class = self.adapted_nn.predict(np.array([self.factual_oh]))[0]
                cf_class = self.adapted_nn.predict(np.array([cf_out_oh]))[0]
            else:
                factual_class = self.adapted_nn.predict(np.array([self.factual_oh]))[0]
                cf_class = self.adapted_nn.predict(np.array([cf_out_oh]))[0]

            if self.output_number == 2:
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
            if verbose:
                if cf_class >= 0.5:
                    logging.info(f'Counterfactual found!\nFactual class:{factual_class}\nCF class:{cf_class}\n')
                else:
                    logging.info(f'Failed counterfactual!\nFactual class:{factual_class}\nCF class:{cf_class}\n')
            return True, cf_out_oh
        else:
            if verbose:
                logging.info(f'No returned counterfactual candidate!\n')
            return False, [np.NaN] * (len(self.factual_oh))
