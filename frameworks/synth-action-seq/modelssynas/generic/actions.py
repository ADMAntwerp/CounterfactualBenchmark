import tensorflow as tf
import numpy as np
from actions.action import Action
import actions.condition_ops as cond

class ChangeCategorical(Action):
    def __init__(self, features, name, target_features_value):
        self.target_features_value = target_features_value
        super().__init__(name=name+' Random Change',
                         description='Change randomly a categorical feature',
                         type='Categoric',
                         num_params=0,
                         features=features,
                         target_features=[self.target_features_value],
                         init_p=[])

    def apply(self, instance, p, use_tensor=True):
        return self.features[self.target_features_value].change_feature_random(instance, 1, use_tensor)

    def get_cost(self, instance, new_instance, use_tensor=True):
        return self.return_cost(instance, new_instance, 5., use_tensor)


class ChangeValue(Action):
    def __init__(self, features, name, target_features_value):
        self.target_features_value = target_features_value
        super().__init__(name=name + 'Change Value',
                         description='Changes Requested Loan Amount',
                         type='Numeric',
                         features=features,
                         target_features=[self.target_features_value],
                         init_p=[0])

    def apply(self, instance, p, use_tensor=True):
        param = self.get_param(p, use_tensor)
        return self.features[self.target_features_value].change_feature_value(instance, param, use_tensor)

    def get_cost(self, instance, new_instance, use_tensor=True):
        new_value = self.features[self.target_features_value].get_feature_value(new_instance, use_tensor, space='x')
        old_value = self.features[self.target_features_value].get_feature_value(instance, use_tensor, space='x')
        change = (new_value - old_value) / old_value
        cost = tf.reduce_sum(tf.square(change)) if use_tensor else np.square(change)
        return self.return_cost(instance, new_instance, cost, use_tensor)

def get_actions(features):

    actions = []

    for c_name, feat in features.items():
        if str(type(feat)).split('.')[-1].split('\'')[0] == 'CategoricFeature':
            actions.append(ChangeCategorical(features, c_name, c_name))
        if str(type(feat)).split('.')[-1].split('\'')[0] == 'Feature':
            actions.append(ChangeValue(features, c_name, c_name))

    return actions
