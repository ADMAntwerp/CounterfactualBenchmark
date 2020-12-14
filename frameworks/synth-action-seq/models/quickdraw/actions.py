import tensorflow as tf
import numpy as np
from actions.action import Action
import actions.condition_ops as cond
import random
import math
# 58, 96, 208, 223, 317, 342, 361, 538, 592, 656, 711, 846, 851, 864, 883, 893


class AddStroke(Action):
    def __init__(self, features):
        x_center = random.uniform(-0.7, 0.7)
        y_center = random.uniform(-0.7, 0.7)
        deg = random.uniform(-math.pi, math.pi)
        length = max(min(random.normalvariate(mu=0.35, sigma=0.1), 0.59), 0.11) / 2
        y_intercept = length * math.sin(deg)
        x_intercept = length * math.cos(deg)
        init_p = [[
            x_center - x_intercept,
            y_center - y_intercept,
            x_center + x_intercept,
            y_center + y_intercept,
        ]]
        super().__init__(name='AddStroke',
                         description='Add a single line',
                         type='Numeric',
                         features=features,
                         target_features=[127],
                         init_p=init_p)
        self.target_stroke = 127

    def set_action(self, start_idx, num_params, target_stroke, sample_stroke, add_tensor=True, rand=False):
        if rand:
            x_center = random.uniform(-0.7, 0.7)
            y_center = random.uniform(-0.7, 0.7)
        else:
            x_center = max(min((sample_stroke[0] + sample_stroke[2]) / 2, 0.7), -0.7)
            y_center = max(min((sample_stroke[1] + sample_stroke[3]) / 2, 0.7), -0.7)
        deg = random.uniform(-math.pi, math.pi)
        length = max(min(random.normalvariate(mu=0.2, sigma=0.05), 0.29), 0.055)
        y_intercept = length * math.sin(deg)
        x_intercept = length * math.cos(deg)
        init_p = [[
            x_center - x_intercept,
            y_center - y_intercept,
            x_center + x_intercept,
            y_center + y_intercept,
        ]]
        self.init_p = init_p
        self.target_stroke = target_stroke
        self.target_features = [target_stroke]
        return self.set_p_selector(start_idx, num_params, add_tensor)

    def apply(self, instance, p, use_tensor=True):
        params = tf.reduce_sum(p * self.p_select_tf[0], 0) if use_tensor else p[self.p_idxs[0]]
        params_tan = tf.tanh(params) if use_tensor else np.tanh(params)
        stroke = (params_tan + 1) / 2 if use_tensor else (params_tan + 1) / 2
        return self.features[self.target_stroke].change_feature_value(instance, stroke, use_tensor)

    def get_cost(self, instance, new_instance, use_tensor=True):
        d = self.features[self.target_stroke].get_feature_value(new_instance, use_tensor)
        return self.return_cost(instance, new_instance, d + 1., use_tensor)

    def postcondition(self, instance, new_instance, use_tensor=True):
        d = self.features[self.target_stroke].get_feature_value(new_instance, use_tensor)
        return cond.op_and(cond.op_gt(d, 0.1, use_tensor, scale=1.),
                           cond.op_lt(d, 1.0, use_tensor, scale=1.),
                           use_tensor)


actions = [
    AddStroke
]
