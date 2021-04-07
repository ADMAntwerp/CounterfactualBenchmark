import pandas as pd
from cfgenerator.data.one_hot_cat import return_encoded


def format_data(data, labels, cat_feats=None, one_hot=False):

    # Convert to DF if Series
    df_data = data if type(data) == pd.DataFrame else pd.Series.to_frame(data).transpose()

    # One-hot encode the categorical features
    if one_hot:
        df_data = return_encoded(df_data, cat_feats)

    return [df_data, pd.Series(labels)]

