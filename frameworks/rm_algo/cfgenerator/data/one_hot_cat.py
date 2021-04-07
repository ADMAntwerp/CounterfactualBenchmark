import pandas as pd
import numpy as np


def _num_detect(df_num):
    for dcat, dtype in df_num.dtypes.to_dict().items():
        if not np.issubdtype(dtype, np.number):
            print(f"WARNING: {dcat} was detect as being {dtype}, not numeric")


def return_encoded(df, cat_feats=None):
    if cat_feats is not None:
        df_cat = df[cat_feats]
        df_num = df.drop(columns=cat_feats)
        _num_detect(df_num)
        df_cat = pd.get_dummies(df_cat)
        return pd.concat([df_num, df_cat], axis=1, sort=False)
    else:
        _num_detect(df)
        return df

