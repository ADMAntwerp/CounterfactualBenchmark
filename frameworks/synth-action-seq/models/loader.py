from actions.feature import loader as feature_loader
from common.paths import GERMAN_DIR, ADULT_DIR, FANNIEMAE_DIR, QUICKDRAW_DIR
from models.german.model import GermanCredit
from models.german.dataset import GermanDataset
from models.german.actions import actions as german_actions
from models.german.model import TRUE_LABEL as german_true_label

from models.adult.model import Adult
from models.adult.dataset import AdultDataset
from models.adult.actions import actions as adult_actions
from models.adult.model import TRUE_LABEL as adult_true_label

from models.fanniemae.model import FannieMae
from models.fanniemae.dataset import FannieMaeDataset
from models.fanniemae.actions import actions as fanniemae_actions
from models.fanniemae.model import TRUE_LABEL as fanniemae_true_label

from models.quickdraw.model import QuickDraw
from models.quickdraw.dataset import QuickdrawDataset
from models.quickdraw.actions import actions as quickdraw_actions
from models.quickdraw.model import TRUE_LABEL as quickdraw_true_label


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