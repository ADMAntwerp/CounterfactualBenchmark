import torch
from mlexplain.sce.abstract_sce import AbstractSCE


class TabularSCE(AbstractSCE):
    def __init__(self, model, target_class):
        super(TabularSCE, self).__init__(model, target_class)

    def init_masks(self, X):
        return torch.zeros_like(X, requires_grad=True)

    def perturbation_operator(self, X, masks):
        return X + masks

    def get_target_class_probabilities(self, X):
        # This implementation assumes that model outputs array of probabilities belonging to class one
        return self.model(X) if self.target_class == 1 else 1.0 - self.model(X)

