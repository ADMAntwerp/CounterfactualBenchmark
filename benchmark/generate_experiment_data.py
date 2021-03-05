"""This module generates all experiments data based on the processed data from
 the External Dataset Treatment notebook and from the train/validation/test indexes
 from the External Model Generator notebook"""

import pandas as pd

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

    # Define the majority class as 0 and the other classes as 1 (following the same treatment
    # given in the model generation)
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

    # load validation data Indexes
    idxs_validation = pd.read_csv(f'../idxsvalidation/{dsName}.csv')
    df_validation = df.loc[idxs_validation['index'].to_list()].copy()
    if len(df_oh) > 0:
        df_oh_validation = df_oh.loc[idxs_validation['index'].to_list()].copy()

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

    # Save datasets
    df_train.to_csv(f'../experiments_data/{dsName}_TRAINDATASET.csv')
    if len(df_oh) > 0:
        df_oh_train.to_csv(f'../experiments_data/{dsName}_TRAINOHDATASET.csv')
    df_test.to_csv(f'../experiments_data/{dsName}_TESTDATASET.csv')

    if len(df_oh) > 0:
        df_oh_test.to_csv(f'../experiments_data/{dsName}_TESTOHDATASET.csv')
    df_validation.to_csv(f'../experiments_data/{dsName}_VALIDATIONDATASET.csv')

    if len(df_oh) > 0:
        df_oh_validation.to_csv(f'../experiments_data/{dsName}_VALIDATIONOHDATASET.csv')

    for class_n in [0, 1]:
        # For each class

        test_factual_set = df_test[(df_test['output'] == class_n)]
        validation_factual_set = df_validation[(df_validation['output'] == class_n)]
        train_factual_set = df_train[(df_train['output'] == class_n)]

        if test_factual_set.shape[0] >= 100:
            # Test set has sufficient rows
            df_factual = test_factual_set.sample(100, random_state=42).copy()
        else:
            # Test does not have sufficient rows, get additional from validation

            if validation_factual_set.shape[0] >= 100 - test_factual_set.shape[0]:
                additional_validation = validation_factual_set.sample(100 - test_factual_set.shape[0], random_state=42).copy()
            else:
                additional_validation = validation_factual_set.copy()

            df_factual = pd.concat([test_factual_set, additional_validation])

            if df_factual.shape[0] < 100:
                # Validation and test are not sufficient, get from training

                if train_factual_set.shape[0] >= 100 - df_factual.shape[0]:
                    additional_training = train_factual_set.sample(100 - df_factual.shape[0], random_state=42).copy()
                else:
                    additional_training = train_factual_set

                df_factual = pd.concat([df_factual, additional_training])

        # Assert there's no duplicated row
        assert pd.DataFrame(df_factual.index).duplicated().sum() == 0

        # Save factual dataset
        df_factual.to_csv(f'../experiments_data/{dsName}_CFDATASET_{str(class_n)}.csv')
