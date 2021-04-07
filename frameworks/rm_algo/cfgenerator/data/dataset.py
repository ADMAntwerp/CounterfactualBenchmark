from cfgenerator.data.one_hot_cat import return_encoded

def format_dataset(dataset, cat_feats=None, one_hot=False):
    if one_hot:
        return return_encoded(dataset, cat_feats)
    return dataset
