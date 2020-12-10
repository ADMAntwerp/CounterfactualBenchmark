import torch
from torch import optim


class AbstractSCE:
    """
    SCE = Sequence of Counterfactuals Explainer.
    This is an abstract class, which implements framework proposed
    by the thesis TODO[insert name here].
    """
    def __init__(self, model, target_class):
        self.model = model
        self.model.eval()
        self.target_class = target_class

    def mask_norm(self, masks):
        """
        Finds norm of the masks

        Parameters
        ----------
        masks
        torch tensor of size (N, k), where
        N is the number of objects,
        k is dimensionality of one mask.

        Returns
        -------
        Torch tensor of size (N) with norms of every object
        """

        # Return L1 norm by default
        return torch.norm(masks, p=1) / masks.shape[0]

    def mask_distance(self, mask1, mask2):
        """
        Returns distance between 2 masks: mask1 and mask2

        Parameters
        ----------
        mask1
        torch tensor of size (k), where k is the dimensionality of the mask

        mask2
        same as mask1

        Returns
        -------
        Scalar value, distance between two masks
        """

        # Return L2 distance by default
        return torch.dist(mask1, mask2, p=2)

    def mean_distances(self, masks):
        """
        Returns mean of pairwise distances between all the masks

        Parameters
        ----------
        masks
        torch tensor of size (N, k), where
        N is the number of objects,
        k is dimensionality of one mask.

        Returns
        -------
        Scalar value, mean of pairwise distances between all the masks
        """
        N = masks.shape[0]

        if N <= 1:
            return 0.0

        res = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                res += self.mask_distance(masks[i], masks[j])

        return 2 * res / (N * (N - 1))

    def perturbation_operator(self, X, masks):
        """
        Perturbs input objects X using masks. In thesis notation, it's Phi(x, m) applied N times,
        where N is the number of objects.

        Parameters
        ----------
        X
        torch tensor, first dimension has N elements = N objects

        masks
        torch tensor of size (N, k), where
        N is the number of objects,
        k is dimensionality of one mask.

        Returns
        -------
        Set of counterfactuals, which has the same dimensionality as X.
        """

        raise NotImplementedError("perturbation_operator should be implemented.")

    def init_masks(self, X):
        """
        Initializes masks for given set of objects X

        Parameters
        ----------
        X
        torch tensor, first dimension has N elements = N objects

        Returns
        -------
        Initialized masks
        """

        raise NotImplementedError("init_masks should be implemented.")

    def get_target_class_probabilities(self, X):
        """
        Return array of probabilities for target class for every object
        Parameters
        ----------
        X
        torch tensor, first dimension has N elements = N objects

        Returns
        -------
        Array of probailities
        """

        return self.model(X)[:, self.target_class]

    def cost(self, X, lambda_coef, mu_coef, return_all_terms=False):
        counterfactuals = self.perturbation_operator(X, self.masks)

        term1 = torch.mean(self.get_target_class_probabilities(counterfactuals))
        term2 = torch.mean(self.mask_norm(self.masks))
        term3 = self.mean_distances(self.masks)

        cost_value = term1 + lambda_coef * term2 + mu_coef * term3

        return cost_value, term1, lambda_coef * term2, mu_coef * term3 if return_all_terms else cost_value

    def fit(self, X, lambda_coef, mu_coef, lr=0.01, n_iter=100,
            force_masks_init=True, verbose=True, verbose_every_iterations=10):
        self.X = X

        if force_masks_init or not hasattr(self, 'masks'):
            self.masks = self.init_masks(X)

        optimizer = optim.SGD([self.masks], lr=lr)

        for i in range(n_iter):
            optimizer.zero_grad()
            cost_value, term1, term2, term3 = self.cost(X, lambda_coef, mu_coef, return_all_terms=True)
            cost_value.backward()

            if verbose and i % verbose_every_iterations == 0:
                print(f"[{i}/{n_iter}] Cost: {cost_value} [{term1}, {term2}, {term3}]")

            optimizer.step()

        return self

    def explanation(self):
        """
        According to thesis TODO[insert name here]
        Returns means of the masks
        NOTE: this method should be called after self.fit(...)

        Returns
        -------
        Mean of the learned masks, so it's 1D tensor
        """

        return torch.mean(self.masks, dim=(0, ))

    def top_k_features(self, k=10, descending=True):
        explanation = self.explanation()

        return torch.argsort(explanation, descending=descending)[:k]
