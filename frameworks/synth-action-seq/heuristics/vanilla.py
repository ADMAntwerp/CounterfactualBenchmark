import itertools
import copy


class VanillaHeuristics(object):
    def __init__(self, actions, max_length=3):
        self.actions = actions
        self.max_length = max_length
        self.sequences = []

    def setup(self, instance):
        self.sequences = self.generate_sequences(self.actions)

    def complete(self):
        return len(self.sequences) == 0

    def get_next(self):
        return self.sequences.pop(0)

    def update(self, result):
        pass

    def generate_sequences(self, actions):
        sequences = list()
        for i in range(self.max_length):
            seq_size = i + 1
            for combination in itertools.product(actions, repeat=seq_size):
                num_params = sum([action.num_params for action in combination], 0)
                sequence = []
                i = 0
                for action in combination:
                    sequence.append(copy.copy(action).set_p_selector(i, num_params))
                    i += action.num_params
                if not is_dupe(sequence):
                    # print([action.name for action in sequence])
                    sequences.append(sequence)
        return sequences


def is_dupe(sequence):

    s = [a.name for a in sequence]
    return len(s)!=len(list(set(s)))

    for n in range(len(sequence) - 1):
        if sequence[n].name == sequence[n + 1].name:
            return True
    return False
