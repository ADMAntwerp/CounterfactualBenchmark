import numpy as np
import bisect
from scipy.sparse import issparse

class Explainer(object):
    def __init__(self, score_f, default_values, prune=True, omit_default=True):
        """
        Build explainer.
        :param score_f: function, scoring function of the model we want to explain
        :param default_values: array-like, we'll replace missing values with imputation. There are the values to be imputed.
        :param prune: bool, whether to ensure that explanations are irreducible.
        :param omit_default: bool, whether to skip explanations from 'default decision' instances.
        """
        self.score_f = score_f
        self.prune = prune
        self.omit_default = omit_default
        self.def_values = default_values
        if issparse(self.def_values):
            self.def_values = np.array(self.def_values.todense()).flatten()
        self.def_values

    def explain(self, data, thresholds=0.5, max_ite=20, stop_at_first=False, cost_func=None):
        """
        Get explanations.
        :param data: Data
        :param thresholds: integer or array (with same number of rows as Data), decision threshold
        :param max_ite: integer, max number of iterations (per instance)
        :param stop_at_first: bool, whether to stop after finding the first explanation for each instance
        :param cost_func: function, it defines the cost of changing the original instance values to the new instance values 
        :return: List of explanations. Each element is a list that contains the explanations for the observation in the ith position. 
        Each explanation is a list that contains the indices of the features present
        in the explanation.
        """
        all_explanations = []
        if not hasattr(thresholds, "__len__"):
            thresholds = np.full(data.shape[0], thresholds)
        if cost_func is None:
            cost_func = self.default_cost_func
        for obs_i, obs in enumerate(data):
            if issparse(obs):
                obs = np.array(obs.todense())
            obs = obs.flatten()
            threshold = thresholds[obs_i]
            obs = obs.reshape(1, -1)
            original_pred = self.score_f(obs)[0]
            score = original_pred - threshold
            # Get class of the observation
            class_val = 1 if score >= 0 else -1
            # Get relevant features to apply operators
            relevant_f = np.where(obs.flatten() != self.def_values)[0]
            # Set lists of explanations
            explanations = np.zeros((0, relevant_f.size))
            e_list = []
            if class_val == 1 or not self.omit_default:
                # Set first combination with no operators applied
                combs = [np.full(relevant_f.size, False, dtype=bool)]
                # Set list of scores
                scores = [score * class_val]
                sorting_scores = [0]
                for i in range(max_ite):
                    # Check if there are any more explanations
                    if (not combs) or (stop_at_first and len(e_list)):
                        break
                    # Get next combination with the smallest score
                    comb = combs.pop(0)
                    score = scores.pop(0)
                    sorting_scores = sorting_scores[1:]
                    # Add to list of explanations if the class changed
                    if score < 0:
                        if self.prune:
                            comb = self.prune_explanation(obs, comb, relevant_f, threshold)
                        explanations = np.vstack((explanations, comb))
                        e_list.append(relevant_f[comb == 1].tolist())
                    else:
                        # Get possible features to apply operator
                        active_f = np.where(np.logical_not(comb))[0]
                        # Build new possible combinations (one for each operator application)
                        new_combs = np.tile(comb, (active_f.size, 1))
                        new_combs[np.arange(active_f.size), active_f] = True
                        # Remove combinations that are a superset of an explanation.
                        matches = new_combs.dot(explanations.T) - explanations.sum(axis=1)
                        are_superset = np.unique(np.where(matches >= 0)[0])
                        new_combs = np.delete(new_combs, are_superset, axis=0)
                        if new_combs.shape[0] == 0:
                            continue
                        # Predict scores for new combs and add them to list
                        new_obs = np.tile(obs, (new_combs.shape[0], 1))
                        def_value_tiles = np.tile(self.def_values[relevant_f], (new_combs.shape[0], 1))
                        new_obs[:, relevant_f] = np.multiply(1 - new_combs, new_obs[:, relevant_f]) \
                                                 + np.multiply(new_combs, def_value_tiles)
                        new_preds = self.score_f(new_obs)
                        costs = cost_func(obs, new_obs)
                        new_sorting_scores = class_val * (new_preds - original_pred) / costs
                        new_scores = class_val * (new_preds - threshold) 
                        for j, new_sorting_score in enumerate(new_sorting_scores):
                            ix = bisect.bisect(sorting_scores, new_sorting_score)
                            sorting_scores.insert(ix, new_sorting_score)
                            scores.insert(ix, new_scores[j])
                            combs.insert(ix, new_combs[j, :])
            all_explanations.append(e_list)
        return all_explanations

    def prune_explanation(self, obs, explanation, active_f, threshold):
        relevant_f = active_f[explanation]
        # Get number of explanation subsets (excluding all variables and no variables)
        n = 2 ** explanation.sum()
        combinations = range(1, n-1)
        # Remove powers of 2 (i.e., single feature combinations)
        combinations = [comb for comb in combinations if (comb & (comb - 1)) > 0]
        n = len(combinations)
        # Order by number of bits (i.e., try larger combinations first)
        combinations = sorted(combinations, key=lambda x: bin(x).count("1"), reverse=True)
        t_obs = np.matrix(obs, copy=True)
        i = 0
        score = self.score_f(obs)[0] - threshold
        class_val = 1 if score >= 0 else -1
        bits = 1 << np.arange(explanation.sum())
        while i < n:
            c = combinations[i]
            # Set features according to combination and predict
            e_bits = ((c & bits) > 0).astype(int)
            t_obs[:, relevant_f] = np.multiply(1 - e_bits, obs[:, relevant_f]) \
                                   + np.multiply(e_bits, self.def_values[relevant_f])
            score = (self.score_f(t_obs) - threshold)[0] * class_val
            if score < 0:
                # We have a shorter explanation
                explanation = np.in1d(active_f, relevant_f[e_bits == 1])
                # Keep only subsets of the combination that was found
                combinations = [comb for comb in combinations if (comb | c) <= c]
                i = 0
                n = len(combinations)
            i += 1
        return explanation

    def default_cost_func(self, original_obs, new_obs):
        return 1