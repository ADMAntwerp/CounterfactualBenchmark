import os
import time
import logging
import random as python_random

import tensorflow as tf
from tensorflow.keras.models import load_model

from keras import backend

import pandas as pd
import numpy as np

from cfbench.cfg import OHConverter
from cfbench.cfg.common import nn_ohe

from cfbench.dataset_data.constants.var_types import VAR_TYPES

CURRENT_PATH = '/'.join(os.path.abspath(__file__).split('/')[:-1])

TOTAL_FACTUAL = 0
for factual_class_count in ['0', '1']:
    for dsname in VAR_TYPES.keys():
        TOTAL_FACTUAL += pd.read_csv(
            f'{CURRENT_PATH}/dataset_data/experiments_data/{dsname}_CFDATASET_{factual_class_count}.csv').shape[0]

logging.basicConfig(level=logging.INFO)

# Set seeds
np.random.seed(42)
python_random.seed(42)
try:
    tf.random.set_random_seed(42)
except AttributeError:
    tf.random.set_seed(42)
tf.compat.v1.disable_v2_behavior()  # disable TF2 behaviour as alibi code still relies on TF1 constructs




class BenchmarkGenerator:

    def __init__(
            self,
            output_number,
            ds_id_test,
            disable_gpu,
            show_progress,
            initial_idx,
            final_idx,):
        self.output_number = output_number
        self.ds_id_test = ds_id_test
        self.show_progress = show_progress

        self.exp_idx = 0
        self.ds_idx = 0
        self.current_dsName = None
        self.factual_class = 0
        self.current_factual_class = 0
        self.factual_idx = 0
        self.total_factuals = None
        self.save_results = False
        self.start_time = None
        self.times_loop_list = []
        self.initial_idx = initial_idx
        self.final_idx = final_idx

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

        if self.initial_idx > 0:
            for _ in range(self.initial_idx):
                self.__next__()

    def __iter__(self):
        return self

    def __next__(self):

        if self.show_progress:
            # If index is one, start the timer
            if self.factual_idx == 0:
                self.start_time = time.time()
            else:
                self.times_loop_list.append(time.time() - self.start_time)

            if len(self.times_loop_list) % 50 == 0 and len(self.times_loop_list) > 0:
                logging.info(
                    f'Experiment {len(self.times_loop_list)}/{TOTAL_FACTUAL} done | '
                    f'Average time: {round(np.mean(self.times_loop_list), 2)} sec | '
                    f'ETA '
                    f'{self._show_time(np.mean(self.times_loop_list)*(TOTAL_FACTUAL-len(self.times_loop_list)))}')

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

        # Verify experiment index
        if self.exp_idx >= self.final_idx:
            raise StopIteration

        dsName = list(VAR_TYPES.keys())[self.ds_id_test[self.ds_idx]]

        if dsName == self.current_dsName and self.current_factual_class == self.factual_class:
            # Go to next row
            self.factual_idx += 1
        else:
            # Assign new dataset name and factual class
            self.current_dsName = dsName
            self.current_factual_class = self.factual_class

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

            self.adapted_nn = nn_ohe(input_shape, hidden_layers_ws, self.output_number)
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

        self.exp_idx += 1

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

    def cf_evaluator(
            self, cf_out: list, algorithm_name: str, cf_generation_time: float = None,
            verbose: bool = False, save_results: bool = False):
        self.save_results = save_results
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
            cf_found = True
            processed_cf = cf_out_oh
        else:
            if verbose:
                logging.info(f'No returned counterfactual candidate!\n')
            cf_found = False
            processed_cf = [np.NaN] * (len(self.factual_oh))

        # Save result in local folder
        if save_results:
            if not os.path.exists('./cfbench_results/'):
                os.mkdir('./cfbench_results/')
            save_df = [{
                'algorithm_name': algorithm_name,
                'dataset_name': self.current_dsName,
                'factual_idx': self.factual_idx,
                'factual_class': self.factual_class,
                'cf_found': cf_found,
                'cf': processed_cf,
                'factual': self.factual_oh,
                'cf_generation_time': cf_generation_time
            }]
            pd.to_pickle(
                save_df,
                f'./cfbench_results/'
                f'{algorithm_name}_{self.current_dsName}_{self.factual_class}_{self.factual_idx}.pkl')

        return cf_found, processed_cf

    # Function that converts minutes to days, hours, minutes, seconds
    def _convert_time(self, time):
        days = time // (24 * 60 * 60)
        time -= days * 24 * 60 * 60
        hours = time // (60 * 60)
        time -= hours * 60 * 60
        minutes = time // 60
        time -= minutes * 60
        seconds = time
        return int(days), int(hours), int(minutes), seconds

    # Function that shows days if days > 0, hours if days > 0 and hours > 0, minutes if days > 0 and hours > 0 and
    # minutes > 0, seconds otherwise
    def _show_time(self, time):
        days, hours, minutes, seconds = self._convert_time(time)

        if days > 0:
            return '{} days, {} hours, {} minutes, {:0.1f} seconds'.format(days, hours, minutes, seconds)
        elif hours > 0:
            return '{} hours, {} minutes, {:0.1f} seconds'.format(hours, minutes, seconds)
        elif minutes > 0:
            return '{} minutes, {:0.1f} seconds'.format(minutes, seconds)
        else:
            return '{:0.1f} seconds'.format(seconds)
