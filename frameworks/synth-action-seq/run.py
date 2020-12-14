import argparse
import json
import time
import sys,os, copy

import tensorflow as tf
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.paths import RESULTS_DIR
from models.loader import load_model, load_env
from heuristics.loader import load_heuristics
from recourse.utils import get_instance_info
from recourse.search import SequenceSearch, ParamsSearch

from recourse.utils import relu_cost_fn
from recourse.config import base_config


class ModelConfig(object):
    def __init__(self, name, ckpt, data):
        self.name = name
        self.ckpt = ckpt
        self.data = data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-model', dest='model_name', type=str)
    parser.add_argument('--ckpt', default='model.h5',dest='model_ckpt', type=str)
    parser.add_argument('--target-data', default='data.npy',dest='data_filename', type=str)
    parser.add_argument('--mode', dest='mode', default='vanilla', type=str)
    parser.add_argument('--l', dest='length', default=4, type=int)
    parser.add_argument('--actions', dest='action_names', type=str, nargs='+')
    parser.add_argument('--exp-name', default='test', type=str)
    parser.add_argument('--instance-id', default=0, type=int)
    options = parser.parse_args()

    if options.model_name == 'quickdraw' and options.data_filename == 'data.npy':
        options.data_filename = 'data.npz'

    return options


def pick_instance(model, target_env, idx=None, false_only=True):
    dataset, actions, features, desired_label = target_env

    if idx is not None:
        if false_only and not is_false(model, dataset, desired_label, idx):
            return None, None, None
        else:
            return dataset.data[idx], dataset.labels[idx], idx
    else:
        idx = -1
        labels = dataset.labels
        instances = dataset.data
        target_instance = target_label = None
        if false_only:
            found = False
            while not found:
                idx = np.random.randint(0, labels.shape[0])
                if is_false(model, dataset, desired_label, idx):
                    found = True
                    target_label = labels[idx]
                    target_instance = instances[idx]
                    print(idx, target_label, target_instance, model.predict(instances[idx:idx + 1]))
        return target_instance, target_label, idx


def is_false(model, dataset, desired_label, idx):
    labels = dataset.labels
    instances = dataset.data
    labeled_false = np.argmax(labels[idx]) != np.argmax(desired_label)
    predicted_false = np.argmax(desired_label) != np.argmax(
        model.predict(instances[idx:idx + 1])[0])
    return labeled_false and predicted_false


def run_on_instance(options, instance_id, sav_dir=None):
    print(options)

    sav_dir = os.path.join(sav_dir, str(instance_id))
    if not os.path.exists(sav_dir):
        os.makedirs(sav_dir) 

    with tf.Session() as session:
        env = load_env(options.model_name, options.data_filename, used_actions=options.action_names)
        model = load_model(options.model_name, options.model_ckpt)
        instance, label, instance_id = pick_instance(model, env, idx=instance_id)
        if instance_id is None:
        	print('Target label satisfied by original instance, stopping...')

        run_info = start_recording(instance_id, options)
        data, actions, features, target_label = env
        for name, feature in features.items():
            feature.initialize_tf_variables()
        if options.model_name == 'quickdraw':
            actions = [action.set_p_selector(i, len(actions)) for i, action in enumerate(actions)]

        heuristics = load_heuristics(options.mode, actions, model, options.length)
        search = SequenceSearch(model, actions, heuristics, sav_dir=sav_dir, config=base_config)

        if options.model_name == 'quickdraw':
            result = search.find_correction(instance.reshape((1, instance.shape[0], instance.shape[1])),
                                        np.array([target_label]), session)
        else:
            result = search.find_correction(instance.reshape((1, instance.shape[0])),
                                            np.array([target_label]), session)

        out = dict(info=get_instance_info(instance, features),
                   output=result.summary() if result.best_result is not None else 'Not Found')
        return end_recording(run_info, out)
        # return dict(instance=instance)


def run_on_instance(options, instance_id, sav_dir=None):
    if sav_dir is not None:
	    sav_dir = os.path.join(sav_dir, str(instance_id))
	    if not os.path.exists(sav_dir):
	        os.makedirs(sav_dir) 

    with tf.Session() as session:
        env = load_env(options.model_name, options.data_filename, used_actions=options.action_names)
        model = load_model(options.model_name, options.model_ckpt)
        instance, label, instance_id = pick_instance(model, env, idx=instance_id)
        run_info = start_recording(instance_id, options)
        data, actions, features, target_label = env
        for name, feature in features.items():
            feature.initialize_tf_variables()
        if options.model_name == 'quickdraw':
            actions = [action.set_p_selector(i, len(actions)) for i, action in enumerate(actions)]

        heuristics = load_heuristics(options.mode, actions, model, options.length)
        search = SequenceSearch(model, actions, heuristics, sav_dir=sav_dir, config=base_config)

        if options.model_name == 'quickdraw':
            result = search.find_correction(instance.reshape((1, instance.shape[0], instance.shape[1])),
                                        np.array([target_label]), session)
        else:
            result = search.find_correction(instance.reshape((1, instance.shape[0])),
                                            np.array([target_label]), session)

        out = dict(info=get_instance_info(instance, features),
                   output=result.summary() if result.best_result is not None else 'Not Found')
        return end_recording(run_info, out)


def start_recording(target_idx, options):
    info = create_run_info(target_idx, options)
    print(
        "\n\n______________________________________________________________________________________________________")
    print(target_idx, info['start'])
    print('Starting %s run on item %d at %d' % (options.mode, target_idx, info['start']))
    return info


def create_run_info(target_idx, options):
    return dict(idx=target_idx,
                mode=options.mode,
                model=options.model_name,
                start=time.time(),
                end=0,
                time_taken=0,
                success=False)


def end_recording(record, result):
    record['end'] = time.time()
    record['time_taken'] = record['end'] - record['start']
    print('Finished %s run on item %d at %d, time taken: %d' % (
        record['mode'], record['idx'], record['end'], record['time_taken']))
    if result['output'] != 'Not Found':
        result['success'] = True
        print('Run Successful for item %d' % (record['idx']))
    else:
        print('No solution found for item %d' % (record['idx']))
    sys.stdout.flush()

    return {**record, **result}
    

if __name__ == '__main__':
    FLAGS = parse_args()
    
    sav_dir = os.path.join('results',FLAGS.exp_name,FLAGS.model_name)

    if not os.path.exists(sav_dir):
        os.makedirs(sav_dir) 

    json.dump(vars(base_config), open(os.path.join(sav_dir, 'config.json'), 'w'), indent=4)

    for i in range(100):
    	# Pass a sav_dir below to save separate output files for each sequence searched
    	# This is useful for running it in a distributed manner
        a = time.time()
        output = run_on_instance(FLAGS, i, sav_dir=None)
        tf.reset_default_graph()
        print('Time taken for instance', time.time()-a)

        if FLAGS.model_name != 'quickdraw' or output['success']:
            save_filename = os.path.join(sav_dir, ('run_%s.json' % (output['idx'])))
            json.dump(output, open(save_filename, 'w+'), indent=4)