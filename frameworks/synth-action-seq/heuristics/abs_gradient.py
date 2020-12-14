import tensorflow as tf
import numpy as np
import heapq
import copy


class AbsGradHeuristics(object):
    def __init__(self, actions, model, max_length=4):
        self.actions = actions
        self.model = model
        self.target = np.reshape(np.array(model.TRUE_LABEL), (1, len(model.TRUE_LABEL)))
        self.max_length = max_length
        self.done = False
        self.setup_tf()
        self.sequences = []
        self.target_idxs = []
        for action in self.actions:
            idxs = []
            if action.type == 'Numeric':
                idxs = [feature.idx for feature in action.target_features]
            elif action.type == 'Categoric':
                idxs = sum([[feature.idx + i for i in range(len(feature.values))] for feature in
                            action.target_features], [])
            self.target_idxs.append((action, idxs))

    def setup(self, instance):
        p = [(-np.inf, -np.inf, [copy.copy(action).set_p_selector(0, action.num_params)])
             for action, target_idxs in self.target_idxs]
        self.sequences = p[:]

        heapq.heapify(self.sequences)

    def complete(self):
        return len(self.sequences) == 0
        # return self.done

    def setup_tf(self):
        logits = self.model.model.output
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.target, logits=logits)
        self.grads = tf.keras.backend.gradients(loss, self.model.model.input)[0]

    def get_next(self):
        return heapq.heappop(self.sequences)[2]

    def update(self, result):
        cur_instance = result.final_instance2
        if cur_instance is None:
            return

        sequence = result.sequence
        if len(sequence) < self.max_length:
            if cur_instance.shape[0] != 1:
                x = np.reshape(cur_instance, (1, cur_instance.shape[0]))
            else:
                x = cur_instance

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                grads = sess.run(self.grads, feed_dict={self.model.model.input: x})

            grads = np.abs(grads)

            next_sequences = []
            l = sum([action.num_params for action in sequence])
            for new_action, target_idxs in self.target_idxs:
                num_params = l + new_action.num_params
                size = len(sequence) + 1
                score = np.mean(grads[0, target_idxs])
                seq = []
                i = 0
                for action in sequence:
                    seq.append(copy.copy(action).set_p_selector(i, num_params))
                    i += action.num_params
                seq.append(copy.copy(new_action).set_p_selector(i, num_params))
                next_sequences.append((size, score, seq))

            for l, g, seq in next_sequences:
                if not is_dupe(seq):
                    heapq.heappush(self.sequences, (-l, -g, seq))


def is_dupe(sequence):
    for n in range(len(sequence) - 1):
        if sequence[n].name == sequence[n + 1].name:
            return True
    return False
