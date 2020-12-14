import tensorflow as tf
import numpy as np


# template for actions


class Action(object):
    def __init__(self, name, description, type, features, target_features, init_p, num_params=1):
        self.name = name
        self.description = description
        self.type = type
        self.num_params = num_params
        self.features = features
        self.target_features = [features[feature_name] for feature_name in target_features]
        self.init_p = init_p
        self.target_features = []
        self.p_select_tf = []
        self.p_select_np = []
        self.p_idxs = []

    def set_p_selector(self, start_idx, num_params, add_tensor=True):
        self.p_select_tf = []
        self.p_select_np = []
        self.p_idxs = []
        for i in range(self.num_params):
            p_idx = start_idx + i
            if self.name == 'AddStroke':
                np_vec = np.zeros((num_params, 4))
                np_vec[p_idx] = [1., 1., 1., 1.]
            else:
                np_vec = np.zeros(num_params)
                np_vec[p_idx] = 1.
            self.p_idxs.append(p_idx)
            self.p_select_np.append(np_vec)
            if add_tensor:
                self.p_select_tf.append(tf.constant(np_vec, dtype=tf.float32))
        return self

    def apply(self, instance, p, use_tensor=True):
        raise RuntimeError('apply undefined for action class %s' % self.name)

    def get_cost(self, instance, new_instance, use_tensor=True):
        raise RuntimeError('cost undefined for action class %s' % self.name)

    def return_cost(self, instance, new_instance, cost, use_tensor=True):
        if use_tensor:
            return cost + self.check_conditions(instance, new_instance, use_tensor)
        else:
            return cost if self.check_conditions(instance, new_instance, use_tensor) else np.inf

    def check_conditions(self, instance, new_instance, use_tensor=True):
        if use_tensor:
            return self.precondition(instance, use_tensor) + self.postcondition(instance,
                                                                                new_instance,
                                                                                use_tensor)
        else:
            return self.precondition(instance, use_tensor) and self.postcondition(instance,
                                                                                  new_instance,
                                                                                  use_tensor)

    def precondition(self, instance, use_tensor=True):
        return 0. if use_tensor else True

    def postcondition(self, instance, new_instance, use_tensor=True):
        return 0. if use_tensor else True

    def get_param(self, p, use_tensor):
        return self.get_ith_param(p, 0, use_tensor)

    def get_ith_param(self, p, i, use_tensor):
        return tf.reduce_sum(p * self.p_select_tf[i]) if use_tensor else p[self.p_idxs[i]]

    def print(self, p):
        if p is None:
            return 'None'
        else:
            return 'Action: %s(%s)' % (self.name, ','.join([str(p[idx]) for idx in self.p_idxs]))

    def __lt__(self, other):
        return self.name < other.name
