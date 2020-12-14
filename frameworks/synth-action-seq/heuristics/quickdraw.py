import numpy as np
import copy
import random


class QuickDrawDriver(object):
    def __init__(self, actions):
        self.actions = actions
        self.sequences = []
        self.original_num_stroke = 94
        self.current_stroke = 127
        self.max_stroke = 127
        self.tries = 5
        self.instance = None

    def setup(self, instance):
        for i in range(instance[0].shape[0]):
            if np.min(instance[0][i]) == np.max(instance[0][i]):
                self.current_stroke = i
                self.max_stroke = min(self.max_stroke, i + 5, 96)
                break
        print('Starting at stroke %d' % self.current_stroke)
        self.original_num_stroke = self.current_stroke - 1
        self.instance = instance
        sample_stroke = self.instance[0][random.randint(0, self.original_num_stroke - 1)]
        self.sequences = [[copy.copy(self.actions[0].set_action(0, 1, self.current_stroke,
                                                                sample_stroke))]]

    def complete(self):
        return len(self.sequences) == 0

    def get_next(self):
        return self.sequences.pop(0)

    def update(self, result):
        l = sum([action.num_params for action in result.sequence], 0)
        if self.tries > 0 and result.cost is None:
            new_sequence = []
            i = 0
            for action in result.sequence:
                sample_stroke = self.instance[0][random.randint(0, self.original_num_stroke - 1)]
                new_sequence.append(copy.copy(action).set_action(i, l,
                                                                 self.current_stroke - len(
                                                                     result.sequence) + 1 + i,
                                                                 sample_stroke))
                i += action.num_params
            self.sequences.append(new_sequence)
            self.tries -= 1
        elif self.current_stroke < self.max_stroke:
            self.tries = 5
            next_action = self.actions[0]
            num_params = l + next_action.num_params
            new_sequence = []
            i = 0
            for action in result.sequence:
                sample_stroke = self.instance[0][random.randint(0, self.original_num_stroke - 1)]
                new_sequence.append(copy.copy(action).set_action(i, num_params,
                                                                 self.current_stroke - len(
                                                                     result.sequence) + 1 + i,
                                                                 sample_stroke))
                i += action.num_params
            sample_stroke = self.instance[0][random.randint(0, self.original_num_stroke - 1)]
            new_sequence.append(
                copy.copy(next_action).set_action(i, num_params, self.current_stroke + 1,
                                                  sample_stroke))
            self.sequences.append(new_sequence)
            self.current_stroke += 1
