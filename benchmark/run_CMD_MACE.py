import sys
# from subprocess import STDOUT, check_output
import subprocess
sys.path.append('../')

import pandas as pd
import numpy as np

from constants.var_types import VAR_TYPES

valid_runs = []

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

    for c in list(df['output'].unique()):

        try:

            cf = pd.read_csv(f'../experiments_data/{str(int(c))}_{dsName}_CFDATASET.csv').drop(columns=['Unnamed: 0'])

            col_number = (df.shape[1] if len(cat_feats) == 0 else  df_oh.shape[1]) - 1
            pd.DataFrame([0] * col_number).T.iloc[:0].to_csv(f'../cfoutput/{str(int(c))}_{dsName}_MACE.csv', index=False)

            for idx_cf in range(cf.shape[0]):
                p = subprocess.Popen(["python3", "./run_experiments_MACE_CMD.py", f"-d {dsName}", f"-c {c}", f"-i {idx_cf}"])
                try:
                    p.wait(1800)
                except subprocess.TimeoutExpired:
                    print("process reached timeout")
                    pd.DataFrame([[np.NaN] * col_number]).to_csv(f'../cfoutput/{str(int(c))}_{dsName}_MACE.csv',
                        mode='a', index=False, header=False)
                    p.kill()
                n_cfs = pd.read_csv(f'../cfoutput/{str(int(c))}_{dsName}_MACE.csv').shape[0]
                if n_cfs < idx_cf + 1:
                    print('Error in cf generation detected')
                    pd.DataFrame([[np.NaN] * col_number]).to_csv(f'../cfoutput/{str(int(c))}_{dsName}_MACE.csv',
                        mode='a', index=False, header=False)

        except FileNotFoundError:
            print(f'{str(int(c))}_{dsName} has no data')