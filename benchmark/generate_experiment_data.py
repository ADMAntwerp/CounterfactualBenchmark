import pandas as pd
import numpy as np

from tensorflow.keras.models import load_model

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

    # Define the majority class as 0 and the other classes as 1
    most_common_class = df['output'].value_counts().index[0]
    df['output'] = df['output'].apply(lambda x: 1 if x == most_common_class else 0)

    df_y_original = df['output'].copy()
    df_oh_y_original = df['output'].copy()

    # Rename classes of full data
    df['output'] = df_y_original.copy()

    if len(df_oh) > 0:
        df_oh['output'] = df_oh_y_original.copy()

    # Load Train data Indexes
    idxs_train = pd.read_csv(f'../idxstrain/{dsName}.csv')
    # Create train data sets
    df_train = df.loc[idxs_train['index'].to_list()].copy()
    if len(df_oh) > 0:
        df_oh_train = df_oh.loc[idxs_train['index'].to_list()].copy()

    # load test data Indexes
    idxs_test = pd.read_csv(f'../idxstest/{dsName}.csv')
    df_test = df.loc[idxs_test['index'].to_list()].copy()
    if len(df_oh) > 0:
        df_oh_test = df_oh.loc[idxs_test['index'].to_list()].copy()

    # Load model
    # model = CFmodel.CreateModel(dsName, str(int(c)))
    model = load_model(f'../models/{dsName}.h5')

    # Sample 100 CF for each category in the DS
    if len(df_oh) > 0:
        df_model = df_oh
        df_model_test = df_oh_test
    else:
        df_model = df
        df_model_test = df_test

    test_pred = model.predict(df_model_test.drop(columns=['output']))
    test_pred = np.apply_along_axis(np.argmax, 1, test_pred)

    full_pred = model.predict(df_model.drop(columns=['output']))
    full_pred = np.apply_along_axis(np.argmax, 1, full_pred)

    # For the 0 class
    test_cf_set = df_test[(df_test['output'] == test_pred) & (df_test['output'] == 0)]
    total_cf_set = df[(df['output'] == full_pred) & (df['output'] == 0)]

    if test_cf_set.shape[0] >= 100:
        df_cf_0 = test_cf_set.sample(100, random_state=42).copy()
    else:
        matches = total_cf_set
        if matches.shape[0] > 0:
            df_cf_0 = matches.sample(100 if matches.shape[0] >= 100 else matches.shape[0], random_state=42).copy()
        else:
            print(f'The dataset {dsName} does not have cf output to be tested')
            df_cf_0 = pd.DataFrame([])

    # For the 1 class
    test_cf_set = df_test[(df_test['output'] == test_pred) & (df_test['output'] == 1)]
    total_cf_set = df[(df['output'] == full_pred) & (df['output'] == 1)]

    if test_cf_set.shape[0] >= 100:
        df_cf_1 = test_cf_set.sample(100, random_state=42).copy()
    else:
        matches = total_cf_set
        if matches.shape[0] > 0:
            df_cf_1 = matches.sample(100 if matches.shape[0] >= 100 else matches.shape[0], random_state=42).copy()
        else:
            print(f'The dataset {dsName} does not have cf output to be tested')
            df_cf_1 = pd.DataFrame([])

    # Save datasets
    df_train.to_csv(f'../experiments_data/{dsName}_TRAINDATASET.csv')
    if len(df_oh) > 0:
        df_oh_train.to_csv(f'../experiments_data/{dsName}_TRAINOHDATASET.csv')
    df_test.to_csv(f'../experiments_data/{dsName}_TESTDATASET.csv')
    if len(df_oh) > 0:
        df_oh_test.to_csv(f'../experiments_data/{dsName}_TESTOHDATASET.csv')
    df_cf_0.to_csv(f'../experiments_data/{dsName}_CFDATASET_0.csv')
    df_cf_1.to_csv(f'../experiments_data/{dsName}_CFDATASET_1.csv')

