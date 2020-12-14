import json
import tensorflow as tf
import numpy as np


class BaseFeature(object):
    def __init__(self, name, idx):
        self.name = name
        self.idx = idx

    def get_feature_value(self, instance, use_tensor, space='z'):
        raise RuntimeError('get_feature_value undefined for %s' % self.name)

    def change_feature_value(self, instance, param, use_tensor):
        raise RuntimeError('change_feature_value undefined for %s' % self.name)

    def print_value(self, instance):
        raise RuntimeError('print_value undefined for %s' % self.name)


class Feature(BaseFeature):
    def __init__(self, name, idx, mean, std, input_dim):
        super().__init__(name=name,
                         idx=idx)
        self.mean = mean
        self.std = std
        self.selector_np = np.zeros(input_dim)
        self.selector_np[idx] = 1.
        self.selector_tf = None

    def initialize_tf_variables(self):
        self.selector_tf = tf.constant(self.selector_np, dtype=tf.float32)

    def get_feature_value(self, instance, use_tensor, space='z'):
        if use_tensor:
            z_val = tf.reduce_sum(instance * self.selector_tf)
        else:
            z_val = instance[self.idx]
        return z_val if space == 'z' else self.ztox(z_val)

    def change_feature_value(self, instance, param, use_tensor):
        selector = self.selector_tf if use_tensor else self.selector_np
        param_vector = param * selector
        return instance + param_vector

    def ztox(self, val, add_mean=True):
        if add_mean:
            return val * self.std + self.mean
        else:
            return val * self.std

    def xtoz(self, val, add_mean=True):
        if add_mean:
            return (val - self.mean) / self.std
        else:
            return val / self.std

    def print(self, instance, space='z'):
        value = self.get_feature_value(instance, use_tensor=False, space=space)
        return '%s: %.6f' % (self.name, value)


class CategoricFeature(BaseFeature):
    def __init__(self, name, idx, values, input_dim, value_map=None):
        super().__init__(name, idx)
        self.values = values
        self.selector = []
        self.one_hot_selector = None
        self.index_vector = None
        self.eye = np.eye(input_dim)
        if value_map is None:
            self.value_map = []
            for i in range(len(values)):
                self.value_map.append([0., 1.])

    def initialize_tf_variables(self):
        self.selector = []
        select_all_categories = np.zeros(self.eye.shape[0])
        for i in range(len(self.values)):
            self.selector.append(tf.constant(self.eye[self.idx + i], dtype=tf.float32))
            select_all_categories += self.eye[self.idx + i]
        self.one_hot_selector = tf.constant(select_all_categories, dtype=tf.float32)
        indexes = np.zeros(self.eye.shape[0])
        for i in range(len(self.values)):
            indexes[self.idx + i] = i
        self.index_vector = tf.constant(indexes, dtype=tf.float32)

    def get_feature_value(self, instance, use_tensor, space='z'):
        if use_tensor:
            return tf.reduce_sum(instance * self.one_hot_selector * self.index_vector)
        else:
            feature_vector = []
            if len(self.values) == 1:  # Hack for FannieMae
                try:
                    val = instance[self.idx]
                    return self.value_map[0].index(val)
                except ValueError:
                    return 0
            for i in range(len(self.values)):
                idx = self.idx + i
                val = instance[idx]
                feature_vector.append(self.value_map[i].index(val))
            return np.argmax(feature_vector)

    def change_feature_value(self, instance, value_idx, use_tensor):
        if use_tensor:
            value_remover = -1 * (self.one_hot_selector - 1.)
            value_setter = self.value_map[value_idx][1] * self.selector[value_idx]
            return instance * value_remover + value_setter
        else:
            new_instance = [x for x in instance]
            for i in range(len(self.values)):
                new_instance[self.idx + i] = 1. if i == value_idx else 0.
            return new_instance

    def print_value(self, instance):
        raise RuntimeError('print_value undefined for %s' % self.name)


class StrokeFeature(BaseFeature):
    def __init__(self, name, idx, input_dim):
        super().__init__(name, idx)
        self.selector_np = np.zeros((input_dim, 4))
        self.selector_np[idx][:] = [1., 1., 1., 1.]
        self.selector_tf = None
        self.helper_np = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32)
        self.helper_tf = None

    def initialize_tf_variables(self):
        self.selector_tf = tf.constant(self.selector_np, dtype=tf.float32)
        self.helper_tf = tf.constant([1, 0, 0, 1, -1, 0, 0, -1], shape=[4, 2], dtype=tf.float32)

    def get_feature_value(self, instance, use_tensor, space='z'):
        if use_tensor:
            stroke = tf.reduce_sum(instance * self.selector_tf, 1)
            return tf.sqrt(tf.reduce_sum(tf.square(tf.matmul(stroke, self.helper_tf))))
        else:
            stroke = instance[self.idx]
            interm = np.matmul(stroke, self.helper_np)
            interm2 = np.square(interm)
            square_d = np.sum(interm2)
            d = np.sqrt(square_d)
            return d

    def change_feature_value(self, instance, param, use_tensor):
        if use_tensor:
            selector = self.selector_tf
            param_vector = param * selector
            current_vector = instance * selector
        else:
            selector = self.selector_np
            param_vector = param * selector
            current_vector = instance * selector
        return instance - current_vector + param_vector


def loader(feature_file):
    """loader for numeric features
    :param feature_file: json file containing feature meta info
    :return: list of Feature objects
    """
    raw_features = json.loads(open(feature_file, 'r').read())
    features = dict()
    input_dim = sum(
        [feature['num_values'] for feature in raw_features if feature['type'] != 'class'])
    for feature in raw_features:
        if feature['type'] == 'numeric':
            features[feature['name']] = Feature(feature['name'], feature['idx'], feature['mean'],
                                                feature['std'], input_dim)
        elif feature['type'] == 'stroke':
            features[feature['name']] = StrokeFeature(feature['name'], feature['idx'], input_dim)
        elif feature['type'] != 'class':
            features[feature['name']] = CategoricFeature(feature['name'], feature['idx'],
                                                         feature['values'], input_dim)
    return features
