import os
import shutil
import tarfile
import requests


def _download_and_unzip_data(save_dir: str, url: str, file_name: str, folder_name: str) -> None:
    """
    Download and unzip a data file
    :param save_dir: Saving directory
    :param url: URL of the data file
    :param file_name: name of the data file
    :param folder_name: output folder name
    :return:
    """

    if not os.path.exists(f'{save_dir}/{folder_name}'):
        os.mkdir(f'{save_dir}/{folder_name}')

    response_download_data = requests.get(f'{url}/{file_name}', stream=True)

    open(f'{save_dir}/{folder_name}/{file_name}', 'wb').write(response_download_data.content)

    tar = tarfile.open(f'{save_dir}/{folder_name}/{file_name}', "r:xz")
    tar.extractall(f'{save_dir}/{folder_name}')
    tar.close()

    os.remove(f'{save_dir}/{folder_name}/{file_name}')


# Get current script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# Function that count all files, recursively, in a directory
def _count_files_in_dir(dir_path: str) -> int:
    """
    Count all files, recursively, in a directory
    :param dir_path: Directory path
    :return: Number of files in the directory
    """
    return sum([len(files) for r, d, files in os.walk(dir_path)])


# Check if the model folder has all required files by counting them
if os.path.exists(f'{SCRIPT_DIR}/model_data'):
    if _count_files_in_dir(f'{SCRIPT_DIR}/model_data') < 27:
        shutil.rmtree(f'{SCRIPT_DIR}/model_data')

# Verify if data folders are present
if not os.path.exists(f'{SCRIPT_DIR}/model_data'):
    print('Creating model_data folder, this may take a while...')
    _download_and_unzip_data(
        save_dir=f'{SCRIPT_DIR}/',
        url='https://objectstorage.us-ashburn-1.oraclecloud.com/n/idaknh7ztshz/b/cfbenchmark/o',
        file_name='model_data.tar.xz',
        folder_name='model_data')


# Check if the data folder has all required files by counting them
if os.path.exists(f'{SCRIPT_DIR}/dataset_data'):
    if _count_files_in_dir(f'{SCRIPT_DIR}/dataset_data') < 415:
        shutil.rmtree(f'{SCRIPT_DIR}/dataset_data')

# Verify if data folders are present
if not os.path.exists(f'{SCRIPT_DIR}/dataset_data'):
    print('Creating dataset_data folder, this may take a while...')
    _download_and_unzip_data(
        save_dir=f'{SCRIPT_DIR}/',
        url='https://objectstorage.us-ashburn-1.oraclecloud.com/n/idaknh7ztshz/b/cfbenchmark/o',
        file_name='dataset_data.tar.xz',
        folder_name='dataset_data')


from . import cfbench

__all__ = [
    'cfbench'
]
