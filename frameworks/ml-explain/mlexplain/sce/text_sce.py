import torch
from mlexplain.sce.abstract_sce import AbstractSCE


class TextSCE(AbstractSCE):
    def __init__(self, model, target_class, index2word):
        super(TextSCE, self).__init__(model, target_class)
        self.idx2word = index2word

    def init_masks(self, X):
        return torch.zeros_like(X, requires_grad=True)

    def perturbation_operator(self, X, masks):
        return X + masks

    def top_k_words_all_masks(self, k=10, descending=False):
        top_indices = torch.argsort(self.masks, dim=1, descending=descending)[:, :k]
        return [list(map(lambda x: self.idx2word[x.item()], top_idxs)) for top_idxs in top_indices]

    def top_k_words(self, k=10, descending=False):
        top_indices = self.top_k_features(k=k, descending=descending)
        return list(map(lambda x: self.idx2word[x.item()], top_indices))
