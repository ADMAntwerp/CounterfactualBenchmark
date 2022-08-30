import os
import shutil

import pandas as pd

from cfbench.benchmark_template_row_instance import BenchmarkGenerator
from cfbench.dataset_data.constants.var_types import VAR_TYPES

CURRENT_PATH = '/'.join(os.path.abspath(__file__).split('/')[:-1])

total_factual = 0
for factual_class in ['0', '1']:
    for dsname in VAR_TYPES.keys():
        test_df = pd.read_csv(
            f'{CURRENT_PATH}/dataset_data/experiments_data/{dsname}_CFDATASET_{factual_class}.csv')
        total_factual += pd.read_csv(
            f'{CURRENT_PATH}/dataset_data/experiments_data/{dsname}_CFDATASET_{factual_class}.csv').shape[0]

class BenchmarkCF:

    def __init__(self, output_number: int = 1, disable_gpu: bool = False):
        self.output_number = output_number
        self.disable_gpu = disable_gpu

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
            disable_gpu=self.disable_gpu)


def process_benchmark(algorithm_name: str):
    algorithm_dfs = []
    for f in os.listdir('./cfbench_results'):
        if '_'.join(f.split('_')[:-3]) == algorithm_name:
            algorithm_dfs.append(pd.read_pickle(f'./cfbench_results/{f}')[0])
    if len(algorithm_dfs) != total_factual:
        raise ValueError(f'Number of results ({len(algorithm_dfs)}) '
                         f'is different from the number of factuals ({total_factual})')

    algorithm_df = pd.DataFrame(algorithm_dfs)

    if not os.path.exists(f'./cfbench_results_processed/'):
        os.mkdir(f'./cfbench_results_processed/')

    algorithm_df.to_pickle(f'./cfbench_results_processed/{algorithm_name}.pkl')


def send_results(algorithm_name: str):
    # Create result dataframe if not existent
    if not os.path.exists(f'./cfbench_results_processed/{algorithm_name}.pkl'):
        process_benchmark(algorithm_name)

    # Ask the user to fork the repository
    print('Please, fork the following repository: https://github.com/rmazzine/Ranking-Tabular-CF.git')
    input('Press enter to continue:')

    # Ask the user for the link of the forked repository
    repository_link = input('Please, enter the link of the forked repository: ')

    repository_folder = repository_link.split('/')[-1].split('.git')[0]

    print('Cloning the repository...')
    if os.path.exists(repository_folder):
        shutil.rmtree(repository_folder)
    os.system(f'git clone {repository_link}')

    # Create a new branch with the algorithm_name
    print('Creating a new branch with the algorithm_name...')
    os.system(f'git checkout -b {algorithm_name}')

    # Copy the results to the repository
    print('Copying the results to the repository...')
    if os.path.exists(f'{repository_folder}/results/{algorithm_name}'):
        print('Removing old results...')
        shutil.rmtree(f'{repository_folder}/results/{algorithm_name}')
    else:
        os.mkdir(f'{repository_folder}/results/{algorithm_name}')

    # Copy the result dataframe to the repository
    shutil.copyfile(f'./cfbench_results_processed/{algorithm_name}.pkl',
                    f'{repository_folder}/results/{algorithm_name}/{algorithm_name}.pkl')

    # Push the results to the repository
    print('Pushing the results to the repository...')
    os.system(f'cd {repository_folder}')
    os.system(f'cd {repository_folder} git add .')
    os.system(f'cd {repository_folder} git commit -m "Update results"')
    os.system(f'cd {repository_folder} git push')

    # Make a pull request to the repository
    print('Now, go to your forked repository and make a pull request to the main branch')
    print(f'Your forked repo link: {repository_link}/tree/{algorithm_name}')

    # Remove the repository
    shutil.rmtree(repository_folder)

