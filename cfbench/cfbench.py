import os
import shutil

import numpy as np
import pandas as pd
import scipy.stats as st

import tensorflow as tf
from tensorflow.keras.models import load_model

from cfbench.benchmark_template_row_instance import BenchmarkGenerator, TOTAL_FACTUAL
from cfbench.dataset_data.constants.var_types import VAR_TYPES
from cfbench.cfg.common import nn_ohe
from cfbench.cfg.analysis import l2, md, madd, sparsity, validity_total, \
    check_one_hot_integrity, check_binary_categorical

CURRENT_PATH = '/'.join(os.path.abspath(__file__).split('/')[:-1])

RUN_PATH = os.getcwd()


class BenchmarkCF:

    def __init__(
            self,
            output_number: int = 1,
            disable_gpu: bool = False,
            show_progress: bool = False,
            initial_idx: int = 0,
            final_idx: int = TOTAL_FACTUAL,):
        self.output_number = output_number
        self.disable_gpu = disable_gpu
        self.show_progress = show_progress
        self.initial_idx = initial_idx
        self.final_idx = final_idx

    def create_generator(
            self,
            dataset_idx: list = None):
        # Verify if indexes of dataset_idx are in the valid range (0 - 21)
        if dataset_idx is not None:
            assert all([0 <= idx <= 21 for idx in dataset_idx]), 'Invalid dataset index, must be in range [0, 21]'
        else:
            dataset_idx = [*range(22)]

        return BenchmarkGenerator(
            output_number=self.output_number,
            ds_id_test=dataset_idx,
            disable_gpu=self.disable_gpu,
            show_progress=self.show_progress,
            initial_idx=self.initial_idx,
            final_idx=self.final_idx)


def process_benchmark(algorithm_name: str):
    algorithm_dfs = []
    for f in os.listdir('./cfbench_results'):
        if '_'.join(f.split('_')[:-3]) == algorithm_name:
            algorithm_dfs.append(pd.read_pickle(f'./cfbench_results/{f}')[0])
    if len(algorithm_dfs) != TOTAL_FACTUAL:
        raise ValueError(f'Number of results ({len(algorithm_dfs)}) '
                         f'is different from the number of factuals ({TOTAL_FACTUAL})')

    algorithm_df = pd.DataFrame(algorithm_dfs)

    if not os.path.exists(f'./cfbench_results_processed/'):
        os.mkdir(f'./cfbench_results_processed/')

    algorithm_df.to_pickle(f'./cfbench_results_processed/{algorithm_name}.pkl')


def _calculate_metrics(metric_data):
    metric_data_values = np.concatenate(metric_data.to_numpy())
    metric_data_values_valid = metric_data_values[~np.isnan(metric_data_values)]
    global_metric_mean = metric_data_values_valid.mean().round(2)
    global_metric_error = st.t.interval(
        0.95,
        len(metric_data_values_valid) - 1,
        loc=np.mean(metric_data_values_valid),
        scale=st.sem(metric_data_values_valid))

    return global_metric_mean, [error_value.round(2) for error_value in global_metric_error]


def print_global_analysis(algorithm_name: str):
    if os.path.exists(f'./cfbench_results_processed/{algorithm_name}_analysis.pkl'):
        algorithm_df = pd.read_pickle(f'./cfbench_results_processed/{algorithm_name}_analysis.pkl')
    else:
        raise ValueError(f'No results found for {algorithm_name}, please, run analyze_results({algorithm_name}) first')

    score_coverage_flip_mean, score_coverage_flip_error = _calculate_metrics(algorithm_df['validityFound'])
    score_coverage_mean, score_coverage_error = _calculate_metrics(algorithm_df['validity'])
    score_sparsity_mean, score_sparsity_error = _calculate_metrics(algorithm_df['sparsity'])
    score_l2_mean, score_l2_error = _calculate_metrics(algorithm_df['l2'])
    score_madd_mean, score_madd_error = _calculate_metrics(algorithm_df['MAD'])
    score_md_mean, score_md_error = _calculate_metrics(algorithm_df['MD'])
    cf_generation_time_mean, cf_generation_time_error = _calculate_metrics(algorithm_df['cf_generation_time'])

    data_report = {
        'Global coverage': [score_coverage_mean, score_coverage_error],
        'Global coverage (class flip)': [score_coverage_flip_mean, score_coverage_flip_error],
        'Global sparsity': [score_sparsity_mean, score_sparsity_error],
        'Global L2 distance': [score_l2_mean, score_l2_error],
        'Global Mean Absolute Deviation': [score_madd_mean, score_madd_error],
        'Global Mahalanobis Distance': [score_md_mean, score_md_error],
        'CF generation time': [cf_generation_time_mean, cf_generation_time_error]
    }

    print("{:<60} {:<15}".format(algorithm_name, 'Error (95% C.I.)'))
    print("{:<40} {:<15} {:<15} {:<15}".format('Metric', 'Mean', 'Lower bound', 'Upper bound'))
    for k, v in data_report.items():
        print("{:<40} {:<15} {:<15} {:<15}".format(k, v[0], v[1][0], v[1][1]))


def analyze_results(algorithm_name: str):
    # Create result dataframe if not existent
    process_benchmark(algorithm_name)

    df_all_cf = pd.read_pickle(f'./cfbench_results_processed/{algorithm_name}.pkl')

    df_rows_results = []

    for current_factual_class in ['0', '1']:
        for dsName in list(VAR_TYPES.keys()):

            # Load feature type specifications
            cat_feats = VAR_TYPES[dsName]['categorical']
            num_feats = VAR_TYPES[dsName]['numerical']

            # Load train data
            df_train = pd.read_csv(
                f'{CURRENT_PATH}/dataset_data/experiments_data/{dsName}_TRAINDATASET.csv').drop(
                columns=['Unnamed: 0'])
            # Load test data
            df_test = pd.read_csv(
                f'{CURRENT_PATH}/dataset_data/experiments_data/{dsName}_TESTDATASET.csv').drop(
                columns=['Unnamed: 0'])
            # Load OH if existent
            if cat_feats:
                # Load train data
                df_oh_train = pd.read_csv(
                    f'{CURRENT_PATH}/dataset_data/experiments_data/{dsName}_TRAINOHDATASET.csv').drop(
                    columns=['Unnamed: 0'])
                # Load test data
                df_oh_test = pd.read_csv(
                    f'{CURRENT_PATH}/dataset_data/experiments_data/{dsName}_TESTOHDATASET.csv').drop(
                    columns=['Unnamed: 0'])
            else:
                df_oh_train = df_train.copy()
                df_oh_test = df_test.copy()

            if current_factual_class == 1:
                df_train['output'] = df_train['output'].map({1: 0, 0: 1})
                df_test['output'] = df_test['output'].map({1: 0, 0: 1})

                df_oh_train['output'] = df_oh_train['output'].map({1: 0, 0: 1})
                df_oh_test['output'] = df_oh_test['output'].map({1: 0, 0: 1})

            # Load model
            model_keras = load_model(f'{CURRENT_PATH}/model_data/models/{dsName}.h5', compile=False)

            # Get architecture info and pass to our NN architecture
            input_shape = model_keras.get_weights()[0].shape[0]
            hidden_layers_ws = model_keras.get_weights()[0].shape[1]

            adapted_nn = nn_ohe(input_shape, hidden_layers_ws, 1)
            adapted_nn.build(input_shape=input_shape)

            adapted_nn.layers[1].set_weights(model_keras.layers[0].get_weights())
            adapted_nn.layers[2].set_weights(model_keras.layers[1].get_weights())

            if current_factual_class == 1:
                # If the tested class is the minority, revert output order
                second_layer_w = []
                for w_row in model_keras.layers[1].get_weights()[0]:
                    second_layer_w.append(np.flip(w_row).tolist())

                second_layer_b = np.flip(model_keras.layers[1].get_weights()[1])

                second_layer = [np.array(second_layer_w, dtype=np.float32),
                                np.array(second_layer_b, dtype=np.float32)]
                adapted_nn.layers[2].set_weights(second_layer)

            adapted_nn.layers[3].set_weights(
                [np.array([[0.0], [1.0]], dtype=np.float32), np.array([0.0], dtype=np.float32)])

            df_cf = df_all_cf[
                (df_all_cf['dataset_name'] == dsName) & (df_all_cf['factual_class'] == int(current_factual_class))
            ].copy()

            cf_generation_time = df_cf['cf_generation_time'].tolist()

            df_cf_data = df_cf.set_index('factual_idx').sort_index()

            columns = df_oh_train.drop(columns=['output']).columns
            cat_columns = [col for col in columns if str(int(col.split('_')[0])) in cat_feats]
            num_columns = [col for col in columns if str(int(col.split('_')[0])) in num_feats]

            df_cf = pd.DataFrame(df_cf_data['cf'].to_list())
            df_cf.columns = columns
            df_factual = pd.DataFrame(df_cf_data['factual'].to_list())
            df_factual.columns = columns
            df_factual['output'] = adapted_nn.predict(df_factual.to_numpy()).round().astype(int)

            # Handle CF that could not be found or that are wrong
            df_cf_found, df_fc_found = _get_cleaned_processed(df_cf, df_factual, adapted_nn)

            # Scores
            score_coverage = validity_total(df_cf, df_factual, adapted_nn)
            if len(cat_columns) > 0:
                bin_check = check_binary_categorical(df_cf, cat_columns)
                ohe_check = check_one_hot_integrity(df_cf, cat_columns)
            else:
                bin_check = np.array([True]*len(df_cf))
                ohe_check = np.array([True]*len(df_cf))
            check_integrity = ohe_check & bin_check
            coverage_integrity = score_coverage & check_integrity
            score_sparsity = sparsity(df_cf, df_cf_found, df_fc_found)
            score_madd = madd(df_oh_train, df_cf, num_columns, cat_columns, df_cf_found, df_fc_found)
            score_l2 = l2(df_cf, df_cf_found, df_fc_found)
            score_md = md(df_oh_train, df_cf, df_cf_found, df_fc_found)

            ds_type = 'numerical'
            if len(cat_columns) > 0:
                ds_type = 'categorical'
                if len(num_columns) > 0:
                    ds_type = 'mixed'

            row_data = {
                'dsname': dsName,
                'ds_type': ds_type,
                'factual_class': current_factual_class,
                'framework': algorithm_name,
                'validity': list(coverage_integrity),
                'validityFound': list(score_coverage),
                'sparsity': list(score_sparsity),
                'l2': list(score_l2),
                'RUC': list(bin_check),
                'RMC': list(ohe_check),
                'MAD': list(score_madd),
                'MD': list(score_md),
                'cf_generation_time': cf_generation_time,
            }
            df_rows_results.append(row_data)

    df_results = pd.DataFrame(df_rows_results)
    df_results.to_pickle(f'./cfbench_results_processed/{algorithm_name}_analysis.pkl')
    print_global_analysis(algorithm_name)


def send_results(algorithm_name: str, generate_analysis: bool = True):
    # Create temporary folder
    temp_folder = '/tmp/cfbench_results_temp'
    if not os.path.exists(temp_folder):
        os.mkdir(temp_folder)
    else:
        shutil.rmtree(temp_folder)
        os.mkdir(temp_folder)

    if generate_analysis:
        # Create result dataframe if not existent
        analyze_results(algorithm_name)

    if not os.path.exists(f'./cfbench_results_processed/{algorithm_name}_analysis.pkl'):
        raise Exception(f'Analysis file not found for {algorithm_name}')

    # Ask the user to fork the repository
    print('Please, fork the following repository: https://github.com/rmazzine/Ranking-Tabular-CF.git')
    input('Press enter to continue:')

    # Ask the user for the link of the forked repository
    valid_link = False
    while not valid_link:
        repository_link = input('Please, enter the SSH path of the forked repository: ')
        if not repository_link.startswith('git@github.com:'):
            print('Invalid repository link, it must be the SSH path of the forked repository')
        else:
            valid_link = True

    print('Cloning the repository...')
    os.system(f'git clone {repository_link} {temp_folder}/')

    # Create a new branch with the algorithm_name
    print('Creating a new branch with the algorithm_name...')
    os.system(f'cd {temp_folder} && git checkout -b {algorithm_name}')

    # Copy the results to the repository
    print('Copying the results to the repository...')
    if os.path.exists(f'{temp_folder}/results/{algorithm_name}'):
        print('Removing old results...')
        shutil.rmtree(f'{temp_folder}/results/{algorithm_name}')
    else:
        os.mkdir(f'{temp_folder}/results/{algorithm_name}')

    # Copy the result dataframe to the repository
    shutil.copyfile(f'./cfbench_results_processed/{algorithm_name}_analysis.pkl',
                    f'{temp_folder}/results/{algorithm_name}/{algorithm_name}_analysis.pkl')

    # Push the results to the repository
    print('Pushing the results to the repository...')
    os.system(f'cd {temp_folder} && git add .')
    os.system(f'cd {temp_folder} && git commit -m "Update results"')
    os.system(f'cd {temp_folder} && git push --set-upstream origin {algorithm_name}')

    # Make a pull request to the repository
    print('Now, make a pull request in the following link:')
    repository_path = repository_link.split("git@github.com:")[1].split('.git')[0]
    print(f'https://github.com/{repository_path}/pull/new/{algorithm_name}')

    # Remove the repository
    shutil.rmtree(temp_folder)


# Uses the dataset model to retrieve only the results that are indeed a counterfactual
def _get_correct(df_cf, df_fc, model):
    # Remove all NaN result rows
    df_cf_found, df_fc_found = df_cf.dropna(), df_fc[0==df_cf.isna().sum(axis=1)]

    # If there are non NaN CF results
    if df_cf_found.shape[0] > 0:
        # Predict the label of the CF results
        labels_cf = model.predict(df_cf_found.to_numpy()).round()

        # Verify if the results flipped the classification and return only those that flipped it
        out_cf = df_cf_found[labels_cf.reshape(-1, 1) != df_fc_found['output'].to_numpy().reshape(-1, 1)]
        out_fc = df_fc_found[labels_cf.reshape(-1, 1) != df_fc_found['output'].to_numpy().reshape(-1, 1)]
        return out_cf, out_fc

    # If there's no result, return empty frames
    return df_cf_found.iloc[:0], df_fc.iloc[:0]


# Function to retrieve only the rows that are indeed counterfactuals
def _get_cleaned_processed(df_cf, df_fc, model):
    # Retrieve only the rows that are indeed CFs
    df_cf_c, df_fc_pro_c = _get_correct(df_cf, df_fc, model)

    return df_cf_c, df_fc_pro_c
