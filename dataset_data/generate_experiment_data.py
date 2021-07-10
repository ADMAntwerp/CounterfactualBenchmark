"""This module generates all experiments data based on the processed data from
 the External Dataset Treatment notebook and from the train/validation/test indexes
 from the External Model Generator notebook"""

import pandas as pd
import numpy as np

from tensorflow.keras.models import load_model

from dataset_data.constants.var_types import VAR_TYPES

for dsName in VAR_TYPES.keys():

    # Load feature type specifications
    cat_feats = VAR_TYPES[dsName]['categorical']
    num_feats = VAR_TYPES[dsName]['numerical']


    # Load Dataset
    if cat_feats and num_feats:
        df = pd.read_csv(f'../dataset_data/data/NORM_{dsName}.csv')
        df_oh = pd.read_csv(f'../dataset_data/data/OH_NORM_{dsName}.csv')
    elif cat_feats:
        df = pd.read_csv(f'../dataset_data/data/{dsName}.csv')
        df_oh = pd.read_csv(f'../dataset_data/data/OH_{dsName}.csv')
    else:
        df = pd.read_csv(f'../dataset_data/data/NORM_{dsName}.csv')
        df_oh = []

    # Load model
    model = load_model(f'../model_data/models/{dsName}.h5')

    # Get prediction labels
    if cat_feats:
        predicted_y = np.apply_along_axis(np.argmax, 1, model.predict(df_oh.drop(columns=['output'])))
    else:
        predicted_y = np.apply_along_axis(np.argmax, 1, model.predict(df.drop(columns=['output'])))

    # Substitute the original labels by the predicted labels
    df['output'] = predicted_y

    df_y_original = df['output'].copy()
    df_oh_y_original = df['output'].copy()

    # Rename classes of full data
    df['output'] = df_y_original.copy()

    if len(df_oh) > 0:
        df_oh['output'] = df_oh_y_original.copy()

    # Load Train data Indexes
    idxs_train = pd.read_csv(f'../dataset_data/idxstrain/{dsName}.csv')
    # Create train data sets
    df_train = df.loc[idxs_train['index'].to_list()].copy()
    if len(df_oh) > 0:
        df_oh_train = df_oh.loc[idxs_train['index'].to_list()].copy()

    # load validation data Indexes
    idxs_validation = pd.read_csv(f'../dataset_data/idxsvalidation/{dsName}.csv')
    df_validation = df.loc[idxs_validation['index'].to_list()].copy()
    if len(df_oh) > 0:
        df_oh_validation = df_oh.loc[idxs_validation['index'].to_list()].copy()

    # load test data Indexes
    idxs_test = pd.read_csv(f'../dataset_data/idxstest/{dsName}.csv')
    df_test = df.loc[idxs_test['index'].to_list()].copy()
    if len(df_oh) > 0:
        df_oh_test = df_oh.loc[idxs_test['index'].to_list()].copy()



    # Save datasets
    df_train.to_csv(f'../dataset_data/experiments_data/{dsName}_TRAINDATASET.csv')
    if len(df_oh) > 0:
        df_oh_train.to_csv(f'../dataset_data/experiments_data/{dsName}_TRAINOHDATASET.csv')
    df_test.to_csv(f'../dataset_data/experiments_data/{dsName}_TESTDATASET.csv')

    if len(df_oh) > 0:
        df_oh_test.to_csv(f'../dataset_data/experiments_data/{dsName}_TESTOHDATASET.csv')
    df_validation.to_csv(f'../dataset_data/experiments_data/{dsName}_VALIDATIONDATASET.csv')

    if len(df_oh) > 0:
        df_oh_validation.to_csv(f'../dataset_data/experiments_data/{dsName}_VALIDATIONOHDATASET.csv')

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
        df_factual.to_csv(f'../dataset_data/experiments_data/{dsName}_CFDATASET_{str(class_n)}.csv')
