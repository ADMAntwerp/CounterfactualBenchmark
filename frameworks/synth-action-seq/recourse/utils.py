import tensorflow as tf
from actions.feature import CategoricFeature, Feature


def relu_cost_fn(model_output, target_label, confidence=0):
    z_target = tf.reduce_sum(target_label * model_output, 1)
    z_other = tf.reduce_max((1 - target_label) * model_output, 1)
    return tf.maximum(0., z_other - z_target - confidence)


def get_instance_info(instance, features):
    instance_info = dict()
    for name, feature in features.items():
        if isinstance(feature, Feature):
            value = feature.ztox(instance[feature.idx])
            instance_info[name] = value
        elif isinstance(feature, CategoricFeature):
            value = feature.get_feature_value(instance, use_tensor=False)
            instance_info[name] = str(value)
    return instance_info


def print_instance_info(info):
    return '\n'.join(['%s: %s' % (k, v) for k, v in info.items()])
