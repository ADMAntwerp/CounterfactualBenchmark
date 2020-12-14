import math
import heapq
import copy

"""
algorithm(instance, actions)
sequences = [(action, INF) for action in actions] 
final = Result()
while not sequences.is_empty():
    sequence = sequences.pop()
    result = search(instance, sequence)
    final.update(result)
    if len(sequence) < max_length:
        sequences.add([(sequence + action, result.cost) for action in actions])
return final    
"""


class StepFullHeuristics(object):
    def __init__(self, actions, max_length=4):
        self.actions = actions
        self.max_length = max_length
        self.sequences = []

    def setup(self, instance):
        self.sequences = [(0, [copy.copy(action).set_p_selector(0, action.num_params)]) for action
                          in self.actions]
        heapq.heapify(self.sequences)

    def complete(self):
        return len(self.sequences) == 0

    def get_next(self):
        seq = heapq.heappop(self.sequences)[1]
        return seq

    def update(self, result):
        l = sum([action.num_params for action in result.sequence], 0)
        cost = math.inf if result.pristine else result.cost
        if len(result.sequence) < self.max_length:
            for next_action in [a for a in self.actions if a.name != result.sequence[-1].name]:
                new_sequence = []
                i = 0
                num_params = l + next_action.num_params
                for action in result.sequence:
                    new_sequence.append(copy.copy(action).set_p_selector(i, num_params))
                    i += action.num_params
                new_sequence.append(copy.copy(next_action).set_p_selector(i, num_params))
                print('adding ', cost, [a.name for a in new_sequence])
                heapq.heappush(self.sequences, (cost, new_sequence))


def is_dupe(sequence):
    for n in range(len(sequence) - 1):
        if sequence[n].name == sequence[n + 1].name:
            return True
    return False
