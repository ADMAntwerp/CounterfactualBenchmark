import pandas as pd

from actions.feature import loader as feature_loader
from common.paths import GERMAN_DIR, ADULT_DIR, FANNIEMAE_DIR, QUICKDRAW_DIR
from modelssynas.german.model import GermanCredit
from modelssynas.german.dataset import GermanDataset
from modelssynas.german.actions import actions as german_actions
from modelssynas.german.model import TRUE_LABEL as german_true_label

from modelssynas.adult.model import Adult
from modelssynas.adult.dataset import AdultDataset
from modelssynas.adult.actions import actions as adult_actions
from modelssynas.adult.model import TRUE_LABEL as adult_true_label

from modelssynas.fanniemae.model import FannieMae
from modelssynas.fanniemae.dataset import FannieMaeDataset
from modelssynas.fanniemae.actions import actions as fanniemae_actions
from modelssynas.fanniemae.model import TRUE_LABEL as fanniemae_true_label

from modelssynas.quickdraw.model import QuickDraw
from modelssynas.quickdraw.dataset import QuickdrawDataset
from modelssynas.quickdraw.actions import actions as quickdraw_actions
from modelssynas.quickdraw.model import TRUE_LABEL as quickdraw_true_label

from modelssynas.generic.actions import get_actions as get_actions_generic


def load_model(name, ckpt):
    if name == 'german':
        return load_german_model(ckpt)
    elif name == 'adult':
        return load_adult_model(ckpt)
    elif name == 'fanniemae':
        return load_fanniemae_model(ckpt)
    elif name == 'quickdraw':
        return load_quickdraw_model(ckpt)


def load_env(name, target_data, used_actions=None):
    if name == 'german':
        return setup_german(target_data, used_actions)
    elif name == 'adult':
        return setup_adult(target_data, used_actions)
    elif name == 'fanniemae':
        return setup_fanniemae(target_data, used_actions)
    elif name == 'quickdraw':
        return setup_quickdraw(target_data, used_actions)


def load_german_model(ckpt_file):
    ckpt_filepath = GERMAN_DIR / ckpt_file
    model = GermanCredit()
    model.restore(str(ckpt_filepath))
    return model


def load_adult_model(ckpt_file):
    ckpt_filepath = ADULT_DIR / ckpt_file
    model = Adult()
    model.restore(str(ckpt_filepath))
    return model


def load_fanniemae_model(ckpt_file):
    ckpt_filepath = FANNIEMAE_DIR / ckpt_file
    model = FannieMae()
    model.restore(str(ckpt_filepath))
    return model


def load_quickdraw_model(ckpt_file):
    ckpt_filepath = QUICKDRAW_DIR / ckpt_file
    model = QuickDraw()
    model.restore(str(ckpt_filepath))
    return model

class generic_dataset:

    def __init__(self, dataset):
        self.data = dataset.drop(columns=['output']).to_numpy()
        self.labels = pd.concat([dataset['output'], abs(dataset['output']-1)], axis=1).to_numpy()

def setup_generic(dataset, cat_feats, num_feats, bin_feats, dict_feat_idx, used_actions=None):

    raw_features = []
    for idx in range(dataset.shape[1]):
        if str(idx) in num_feats:
            input_data = {'type': 'numeric',
                          'name': str(idx),
                          'idx': list(dataset.columns).index(str(idx)),
                          'i': idx,
                          'num_values': 1,
                          'mean': dataset[str(idx)].mean(),
                          'std': dataset[str(idx)].std()
                          }
            raw_features.append(input_data)
        if str(idx) in cat_feats:
            input_data = {'type': 'nominal',
                          'name': str(idx),
                          'values': [1]*len(dict_feat_idx[str(idx)].keys()),
                          'idx': min(list(dict_feat_idx[str(idx)].values())),
                          'i': idx,
                          'num_values': len(dict_feat_idx[str(idx)].keys())}

            raw_features.append(input_data)

    features = feature_loader('', raw_features)
    actions = get_actions_generic(features)

    dataset = generic_dataset(dataset)

    return dataset, actions, features, [0.0, 1.0]

def setup_german(target_data, used_actions=None):
    data_filepath = GERMAN_DIR / target_data
    feat_filepath = GERMAN_DIR / 'german.features.json'

    dataset = GermanDataset(str(data_filepath))
    features = feature_loader(str(feat_filepath))
    actions = [action_cls(features) for action_cls in german_actions]
    actions = [action for action in actions if used_actions is None or action.name in used_actions]

    return dataset, actions, features, german_true_label


def setup_adult(target_data, used_actions=None):
    data_filepath = ADULT_DIR / target_data
    feat_filepath = ADULT_DIR / 'adult.features.json'

    dataset = AdultDataset(str(data_filepath))
    features = feature_loader(str(feat_filepath))
    actions = [action_cls(features) for action_cls in adult_actions]
    actions = [action for action in actions if used_actions is None or action.name in used_actions]
    return dataset, actions, features, adult_true_label


def setup_fanniemae(target_data, used_actions=None):
    data_filepath = FANNIEMAE_DIR / target_data
    feat_filepath = FANNIEMAE_DIR / 'fanniemae.features.json'

    dataset = FannieMaeDataset(str(data_filepath))
    features = feature_loader(str(feat_filepath))
    actions = [action_cls(features) for action_cls in fanniemae_actions]
    actions = [action for action in actions if used_actions is None or action.name in used_actions]
    return dataset, actions, features, fanniemae_true_label


def setup_quickdraw(target_data, used_actions=None):
    data_filepath = QUICKDRAW_DIR / target_data
    feat_filepath = QUICKDRAW_DIR / 'cat.features.json'

    dataset = QuickdrawDataset(str(data_filepath))
    features = feature_loader(str(feat_filepath))
    actions = [action_cls(features) for action_cls in quickdraw_actions]
    actions = [action for action in actions if used_actions is None or action.name in used_actions]
    return dataset, actions, features, quickdraw_true_label