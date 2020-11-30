import sys

sys.path.append('../frameworks/DiCE/')

import dice_ml

def get_cf(df_train, num_feats, cat_feats, model, cf):

    if cat_feats:
        d = dice_ml.Data(dataframe=df_train,
                         continuous_features=num_feats,
                         outcome_name='output')
        m = dice_ml.Model(model)
        exp = dice_ml.Dice(d, m)
        converted_cf = {}
        converted_cf_list = []
        for k, v in cf.drop(columns=['output']).to_dict().items():
            converted_cf[str(k)] = str(v)
            converted_cf_list.append(str(v))
        cfs = exp.generate_counterfactuals(
            converted_cf_list,
            total_CFs=1,
            desired_class="opposite")
        return cfs
    else:
        d = dice_ml.Data(dataframe=df_train,
                         continuous_features=num_feats,
                         outcome_name='output')
        m = dice_ml.Model(model)
        exp = dice_ml.Dice(d, m)
        cfs = exp.generate_counterfactuals(
            cf.drop(columns=['output']).to_dict(),
            total_CFs=1,
            desired_class="opposite")
        return cfs
