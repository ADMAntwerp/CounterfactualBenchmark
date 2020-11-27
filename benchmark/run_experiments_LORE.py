import time
import sys
sys.path.append('../frameworks/LORE/')

import pandas as pd
import numpy as np

from benchmark.cfg import CFmodel
from benchmark.cfg import OHConverter

from frameworks.LORE import test_lore

from constants.var_types import VAR_TYPES



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

        # Rename classes of full data
        df['output'] = df_y_original.copy()
        df['output'] = df['output'].apply(lambda x: 0 if x == c else 1)
        if len(df_oh) > 0:
            df_oh['output'] = df_oh_y_original.copy()
            df_oh['output'] = df_oh['output'].apply(lambda x: 0 if x == c else 1)

        # Load Train data Indexes
        idxs_train = pd.read_csv(f'../idxstrain/{str(int(c))}_{dsName}.csv')
        # Create train data sets
        df_train = df.loc[idxs_train['index'].to_list()].copy()
        if len(df_oh) > 0:
            df_oh_train = df_oh.loc[idxs_train['index'].to_list()].copy()

        # load test data Indexes
        idxs_test = pd.read_csv(f'../idxstest/{str(int(c))}_{dsName}.csv')
        df_test = df.loc[idxs_test['index'].to_list()].copy()
        if len(df_oh) > 0:
            df_oh_test = df_oh.loc[idxs_test['index'].to_list()].copy()

        # Load model
        model = CFmodel.CreateModel(dsName, str(int(c)))

        # Sample 10 CF for each category in the DS
        if len(df_oh) > 0:
            df_model = df_oh
            df_model_test = df_oh_test
        else:
            df_model = df
            df_model_test = df_test

        if df_test[(df_test['output'] == model.predict(df_model_test.drop(columns=['output']))) &
                   (df_test['output'] == 0)].shape[0] >= 10:
            df_cf = df_test[(df_test['output'] == model.predict(df_model_test.drop(columns=['output']))) &
                            (df_test['output'] == 0)].sample(10).copy()
        else:
            matches = df[((df['output'] == model.predict(df_model.drop(columns=['output']))))
                       & (df['output'] == 0)]
            if matches.shape[0] > 0:
                df_cf = matches.sample(10 if matches.shape[0] >= 10 else matches.shape[0]).copy()
            else:
                df_cf = pd.DataFrame([])
                print(f'WARNING, THE DATASET {dsName} DOES NOT HAVE ANY CORRECT PREDICTION FOR THE CLASS {str(int(c))}')

        # Save dataset
        df_cf.to_csv(f'../cfoutput/{str(int(c))}_{dsName}_CFDATASET.csv')

        # If the model only outputs one class, it probably will not generate CF classes
        df_test = df_oh if len(df_oh) > 0 else df
        if len(np.unique(model.predict(df_test.drop(columns=['output'])))) > 1:
            # LORE
            timeRunLORE = []
            for idx_cf in range(df_cf.shape[0]):
                start_time = time.time()
                if len(df_oh) > 0:
                    converterLORE = OHConverter.Converter(df, cat_feats, list(df_oh.columns))
                    cfLORE = test_lore.main(df_train, cat_feats, 'output', list(df_oh.columns), model, df_cf.iloc[idx_cf])
                else:
                    converterLORE = OHConverter.Converter(df, cat_feats, list(df.columns))
                    cfLORE = test_lore.main(df_train, cat_feats, 'output', list(df.columns), model, df_cf.iloc[idx_cf])
                pd.DataFrame(converterLORE.convert(cfLORE)).to_csv(f'../cfoutput/{str(idx_cf)}_cf_{str(int(c))}_{dsName}_LORE.csv', index=False)
                end_time = time.time()
                timeRunLORE.append([dsName, end_time-start_time])
                pd.DataFrame(timeRunLORE).to_csv('../cfoutput/TIME_LORE.csv', mode='a')
        else:
            print(f'WARNING, THE MODEL FROM THE DATASET {dsName} AND CLASS {str(int(c))} HAS A MODEL THAT ONLY OUTPUTS'
                  f'ONE CLASS. THEREFORE, IT WILL BE SKIPPED BECAUSE IT MAY NOT GENERATE CF CLASSES')