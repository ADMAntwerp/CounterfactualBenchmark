from collections import defaultdict

import numpy as np
import scipy as sp
import pandas as pd


# Calculate the L2 (euclidean) distance
def l2(df_cf, df_cf_found, df_fc_found):

    # If there's counterfactual explanations
    if df_cf_found.shape[0] > 0:

        # Copy the found CF dataframe
        df_fc_found = df_fc_found.copy().drop(columns=['output'])

        # Create a list to store the results
        scores = []

        # For each cf result row, calculate the L2 distance
        for i in range(df_cf_found.shape[0]):
            scores.append(np.linalg.norm(
                df_cf_found.iloc[i].round(4).to_numpy().reshape(-1, 1) -
                df_fc_found.iloc[i].round(4).to_numpy().reshape(-1, 1)))

        # Change the variable output name to the standard
        out_array = scores

        # Create the output list
        output_results = [np.nan] * df_cf.shape[0]

        # For each index where it was found a CF, insert the result
        for idx_result, idxFound in enumerate(list(df_fc_found.index)):
            output_results[idxFound] = out_array[idx_result]

        return output_results

    return [np.nan] * df_cf.shape[0]


# Function to calculate the sparsity score
def sparsity(df_cf, df_cf_found, df_fc_found):

    # If there's counterfactual result
    if df_cf_found.shape[0] > 0:

        # Copy the found CF results
        df_fc_found = df_fc_found.copy().drop(columns=['output'])

        # Create an array to store output results
        scores = []

        # For each result index
        for i in range(df_cf_found.shape[0]):
            # Calculate the number of changes by the total number of features (sparsity)
            scores.append((df_cf_found.iloc[i].round(4) == df_fc_found.iloc[i].round(4)).sum() / df_cf_found.shape[1])

        # Transfer the result to the standard output variable
        out_array = scores

        # Create the result output array
        output_results = [np.nan] * df_cf.shape[0]

        # For each found result, insert it in the output array
        for idx_result, idxFound in enumerate(list(df_fc_found.index)):
            output_results[idxFound] = out_array[idx_result]

        return output_results

    return [np.nan] * df_cf.shape[0]


# Returns the coverage (validity) score, that inform if a CF indeed flipped the class result
def validity_total(df_cf, df_fc, model):
    # (not_is_na) and ((cf > 0.5)!=(fc > 0.5) or cf==0.5)

    not_is_na = df_cf.isna().sum(axis=1) == 0

    # Although we fill na with 0, we already know in not_is_na the rows that are not na
    cf = pd.Series(model.predict(df_cf.fillna(0)).reshape(-1))
    fc = pd.Series(model.predict(df_fc.drop(columns=['output'])).reshape(-1))

    return (not_is_na & (((cf > 0.5) != (fc > 0.5)) | (cf.apply(lambda x: round(x, 5)) == 0.5))).to_list()


# Calculates the Mean Absolute Deviation Distance
def madd(df_oh, df_cf, num_columns, cat_columns, df_cf_found, df_fc_found):

    # If there are counterfactuals
    if df_cf_found.shape[0] > 0:

        # Create an empty frame to store the results
        df_mad = df_cf_found.iloc[:0]

        # Create a copy of the df_oh frame, avoiding to alterate the original
        df_oh_c = df_oh.copy()

        # Get only the found CFs
        df_oh_c.columns = df_fc_found.columns

        # Create a dictionary to store the MAD distance for each result
        mad_num = {}
        for n_feat_idx in num_columns:
            # 1e-8 added to avoid 0 and, then, division by zero
            mad_num[n_feat_idx] = sp.stats.median_abs_deviation(df_oh_c.loc[:, n_feat_idx]) + 1e-8

            # Calculate the distance using the MAD
            df_mad[n_feat_idx] = abs(df_cf_found[n_feat_idx] - df_fc_found[n_feat_idx]) / mad_num[n_feat_idx]

        for c_feat_idx in cat_columns:
            # If it's a categorical feature, we use 1 distance if it's different and 0 if the same
            df_mad[c_feat_idx] = (df_cf_found[c_feat_idx] != df_fc_found[c_feat_idx]).map(int)

        # Create an array to output the result
        output_result = [0] * df_cf.shape[0]

        # If there are categorical features
        if len(cat_columns) > 0:
            # Get the mean reasult for the categorical features distances
            add_output_result = df_mad[cat_columns].mean(axis=1)

            # For those rows that did not generate CF results
            for null_row in list(set([*range(len(output_result))]) - set(df_mad.index)):
                add_output_result.loc[null_row] = np.nan

            # Sort by the index order
            add_output_result = add_output_result.sort_index()

            # Sum the results to the output array
            output_result = np.add(output_result, add_output_result.tolist())

        # If there are numerical features
        if len(num_columns) > 0:

            # Get the mean reasult for the numerical features distances
            add_output_result = df_mad[num_columns].mean(axis=1)

            # For those rows that did not generate CF results
            for null_row in list(set([*range(len(output_result))]) - set(df_mad.index)):
                add_output_result.loc[null_row] = np.nan

            # Sort by the index order
            add_output_result = add_output_result.sort_index()

            # Sum the results to the output array
            output_result = np.add(output_result, add_output_result.tolist())

        # Convert the output result frame to a list
        out_array = output_result.tolist()

        # Create a final output array with NAN values
        output_results = [np.nan] * df_cf.shape[0]

        # Replace the NaN values for results when there's a found CF result
        for idx_result, idxFound in enumerate(list(df_fc_found.index)):
            output_results[idxFound] = out_array[idx_result]

        return output_results

    return [np.nan] * df_cf.shape[0]


# Mahalanobis Distance metric calculation
def md(df_oh, df_cf, df_cf_found, df_fc_found):

    # If there are counterfactuals
    if df_cf_found.shape[0] > 0:

        # Create array to store the results
        output_result = []

        # For each row index
        for idx in range(df_cf_found.shape[0]):
            # Calculate the mahalanobis distance between the counterfactual and factual and
            # having the dataset covariance matrix
            m_dis = sp.spatial.distance.mahalanobis(df_cf_found.iloc[idx].to_numpy(),
                                                    df_fc_found.drop(columns=['output']).iloc[idx].to_numpy(),
                                                    df_oh.drop(columns=['output']).cov().to_numpy())
            # Append row results
            output_result.append(m_dis)

        # Store the result in another variable to mantain pattern
        out_array = output_result

        # Create a NaN array with the same number of CF results
        output_results = [np.nan] * df_cf.shape[0]

        # Iterate over the FOUND cf result index and order to fill the output result array
        for idx_result, idxFound in enumerate(list(df_fc_found.index)):
            # For the found CF index, insert the found result (calculated mahalanobis distance)
            output_results[idxFound] = out_array[idx_result]

        return output_results

    return [np.nan] * df_cf.shape[0]


# This result check if categorical features follow the binarization rule
def check_binary_categorical(df_cf, cat_columns):
    df_not_nan = df_cf.isna().sum(axis=1) == 0
    df_has_bin_values = ((df_cf.loc[:, cat_columns] != 1) & (df_cf.loc[:, cat_columns] != 0)).sum(axis=1) == 0

    return (df_not_nan & df_has_bin_values).to_numpy()


# Verifies if the one-hot encoded features only activated one feature
def check_one_hot_integrity(df_cf, cat_columns):
    # Group columns with the same prefix
    cat_groups = defaultdict(list)
    for col in cat_columns:
        cat_groups[col.split('_')[0]].append(col)
    df_not_nan = df_cf.isna().sum(axis=1) == 0
    check_groups = [df_cf.loc[:, group_values].sum(axis=1) == 1 for group_values in cat_groups.values()
                    if len(group_values) > 1]
    df_ohe_integrity = sum(check_groups) == len(check_groups)

    return (df_not_nan & df_ohe_integrity).to_numpy()
