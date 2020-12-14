from recourse.utils import relu_cost_fn
from recourse.result import SearchResult, ParamsResult
from common.log import logger

import tensorflow as tf
import numpy as np
import math, json, os

from datetime import datetime

class SequenceSearch(object):
    def __init__(self, model, actions, heuristics, config, sav_dir=None, label_cost_fn=relu_cost_fn):
        self.model = model
        self.actions = actions
        self.driver = heuristics
        self.label_cost_fn = label_cost_fn
        self.config = config
        self.sav_dir = sav_dir

    def find_correction(self, instance, target, sess):
        result = SearchResult(instance)
        self.driver.setup(instance)
        start_time = datetime.utcnow()
        while not self.driver.complete():
            sequence = self.driver.get_next()
            print([action.name for action in sequence])
            if sequence[0].name == 'AddStroke':
                search = ParamsSearch(self.model, sequence, (None, instance.shape[1], instance.shape[2]),
                                      target.shape[1], self.label_cost_fn, self.config, sav_dir=self.sav_dir)
            else:
                search = ParamsSearch(self.model, sequence, (None, instance.shape[1]),
                                      target.shape[1], self.label_cost_fn, self.config, sav_dir=self.sav_dir)
            sequence_result, reached = search.find_params(instance, target, sess)
            self.driver.update(sequence_result)
            result.update(sequence_result)

        if self.sav_dir is not None:
            json.dump(result.summary(), open(os.path.join(self.sav_dir, 'summary.json'),'w'), indent=4)

        return result


class ParamsSearch(object):
    
    def __init__(self, model, sequence, input_shape, label_shape, label_cost_fn, config,
                 batch_size=1, sequence_search='enumerate', sav_dir=None):
        self.sav_dir = sav_dir

        self.init_p = tf.constant(np.array(sum([action.init_p for action in sequence], []) * batch_size),
                                  dtype=tf.float32)
        self.learning_rate = config.learning_rate
        self.adam_beta1 = config.adam_beta1
        self.adam_beta2 = config.adam_beta2
        self.adam_eps = config.adam_eps
        self.starting_cost = config.starting_cost
        self.binary_searches = config.binary_searches
        self.init_c = config.init_c
        self.c_upper_bound = config.c_upper_bound
        self.c_lower_bound = config.c_lower_bound
        self.iterations = config.iterations
        self.abort_early = True
        self.search_search = sequence_search

        self.sequence = sequence
        self.p = tf.Variable(self.init_p, dtype=tf.float32)
        self.instance = tf.placeholder(shape=input_shape, dtype=tf.float32)

        self.target_label = tf.placeholder(shape=(None, label_shape), dtype=tf.float32)
        self.c = tf.placeholder(shape=(1,), dtype=tf.float32)

        self.final_instance, path = self.transform(self.instance, sequence, self.p)
        self.cost = self.get_cost(path, sequence)

        self.output = model(self.final_instance)
        self.f = label_cost_fn(self.output, self.target_label)

        self.loss = tf.reduce_sum(self.cost) + tf.reduce_sum(self.c * self.f)
        if 'Numeric' not in set([action.type for action in sequence]):
            self.minimize = None
            self.init = tf.variables_initializer(var_list=[self.p])
        else:
            optimizer = tf.train.AdamOptimizer(self.learning_rate, self.adam_beta1, self.adam_beta2,
                                               self.adam_eps)
            self.minimize = optimizer.minimize(self.loss, var_list=[self.p])

            self.init = tf.variables_initializer(var_list=[self.p] + optimizer.variables())

    def transform(self, instance, sequence, params):
        path = [instance]
        for i, action in enumerate(sequence):
            path.append(action.apply(path[-1], params))
        return path[-1], path

    def get_cost(self, path, sequence):
        cost = tf.constant(0., dtype=tf.float32)
        for i, action in enumerate(sequence):
            cost += action.get_cost(path[i], path[i+1])
        return cost

    def find_params(self, instance, target, sess):
        result = ParamsResult(self.sequence, instance)

        c = np.array([self.init_c])
        sess.run(self.init)
        sess.run([self.p.assign(self.init_p)])
        if self.minimize is None:
            loss, cost, f, p, final = sess.run(
                [self.loss, self.cost, self.f, self.p, self.final_instance],
                feed_dict={self.instance: instance,
                           self.target_label: target,
                           self.c: c})
            result.update(loss, cost, f[0], p, final[0])
            reached = f[0]==0.
            logger.debug('Final: loss=%.9f cost=%.9f, f=%s, params=%s' % (result.loss, result.cost, result.f, result.p))    
            return result, reached

        check_every = 100
        prev = self.starting_cost
        target_reached = False
        last_f = None

        for iteration in range(self.iterations):
            _, loss, cost, f, p, final = sess.run(
                [self.minimize, self.loss, self.cost, self.f, self.p, self.final_instance],
                feed_dict={self.instance: instance,
                           self.target_label: target,
                           self.c: c})
            if math.isnan(loss):
                print('NaN loss')
                break
            # Update c in minimization objective cost + c * f
            if iteration % check_every == 0:
                if f[0] == 0.:
                    c = max(np.array([1e-5], dtype=np.float32), c / 10)
                    check_every = max(check_every * 2, 10000)
                    target_reached = True
                elif f[0] > 0 and target_reached:
                    c = min(np.array([1e9], dtype=np.float32), c * 2)
                    check_every = max(check_every / 2, 10)
                elif f[0] > 0 and not target_reached:
                    if last_f == f[0]:
                        c = np.array([1], dtype=np.float32)
                    else:
                        c = min(np.array([1e9], dtype=np.float32), c * 2)
                last_f = f[0]

            if iteration % (self.iterations // 100) == 0:
                logger.debug('%d: c=%s, loss=%.9f cost=%.9f, f=%s, params=%s' % (
                    iteration, c[0], loss, cost, f, p))

                if self.abort_early:
                    
                    # print('Checking:', result.success, cost, result.cost, f, result.f)
                    if not result.success and f[0] > 0.9999*result.f:
                        result.update(loss, cost, f[0], p, final[0])
                        break

                    if result.success:
                        if f[0]==0.:
                            if cost > 0.9999*result.cost:
                                result.update(loss, cost, f[0], p, final[0])
                                break
                        else:
                            break

            result.update(loss, cost, f[0], p, final[0])

        logger.debug('Final %d: loss=%.9f cost=%.9f, f=%s, params=%s' % (iteration, result.loss, result.cost, result.f, result.p))    
        if self.sav_dir is not None:
            json.dump(result.summary(), open(os.path.join(self.sav_dir, result.name+'.json'),'w'), indent=4)

        return result, target_reached


class ParamsStrokeSearch(ParamsSearch):
    def __init__(self, model, sequence, input_shape_1, input_shape_2, label_shape, label_cost_fn,
                 config, batch_size=1, sequence_search='enumerate'):
        super().__init__(model, sequence, input_shape_1, label_shape, label_cost_fn, config,
                         batch_size, sequence_search)
        self.instance = tf.placeholder(shape=(None, input_shape_1, input_shape_2), dtype=tf.float32)