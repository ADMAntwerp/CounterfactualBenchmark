"""
Implemented method based on https://arxiv.org/abs/1711.00399
'Counterfactual Explanations without Opening the Black Box: Automated Decisions and the GDPR'
"""

import torch
import numpy as np
import torchvision
from torch.autograd import Variable


class WachterCounterfactual:
    def __init__(self, model, distance_func='euclidean'):
        self.model = model
        self.distance_func = self._get_distance_func(distance_func)

    # TODO (don't do optimization by yourself, use existing optimizer for it!)
    def generate(self, X, target_class, to_maximize=True, threshold=1e-5, grad_step=1.0, n_iters=100, verbose=False):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X)

        def solve_opt_problem(lambda_coef):
            x_var = Variable(X.clone(), requires_grad=True)

            for i in range(n_iters):
                if to_maximize:
                    cost = lambda_coef * (1.0 - self.model(x_var)[0, target_class]) ** 2 \
                           + self.distance_func(X, x_var)
                else:
                    cost = lambda_coef * self.model(x_var)[0, target_class] ** 2 \
                           + self.distance_func(X, x_var)

                if verbose:
                    print("Iteration: {}/{}; Lambda: {} Cost: {}".format(i + 1, n_iters, lambda_coef, cost.item()))
                cost.backward()

                with torch.no_grad():
                    x_var -= grad_step * x_var.grad
                    x_var.grad.zero_()

            return x_var, \
                   torch.abs(1.0 - self.model(x_var)[0, target_class]) \
                       if to_maximize \
                       else torch.abs(self.model(x_var)[0, target_class])

        MAX_ITERATIONS = 20
        lambda_coef = 10.0
        lambda_multiplier = 2.0

        # Iteratively solves for different lambdas, until we reach the desired threshold for target_class
        for i in range(MAX_ITERATIONS):
            x_var, diff = solve_opt_problem(lambda_coef)

            if diff < threshold:
                return x_var

            lambda_coef *= lambda_multiplier

        # In this case we weren't able to find close enough point (in terms of target_class)
        # You could consider increasing the threshold.
        return None

    @staticmethod
    def _get_distance_func(distance_func):
        if callable(distance_func):
            return distance_func

        name_to_distance = {
            'euclidean': torch.dist
        }

        if isinstance(distance_func, str):
            if distance_func not in name_to_distance:
                raise RuntimeError("Distance function '{}' not found. Provide your own or use own of the following "
                                   "types: {}".format(distance_func, list(name_to_distance.keys())))

            return name_to_distance[distance_func]
        else:
            raise RuntimeError("Type {} for distance_func is not supported. Either use name of the function, "
                               "or provide your own function".format(type(distance_func)))


if __name__ == "__main__":
    from mlexplain.vision.utils import plot_img, plot_img_heatmap, load_img, plot_row_images
    from mlexplain.utils import read_imagenet_clsidx_to_labels
    from torch.nn import Softmax
    import torch.nn.functional as F

    clsidx_to_labels = read_imagenet_clsidx_to_labels('data/imagenet_1000_clsidx_to_labels.json')

    model = torchvision.models.vgg16(pretrained=True)
    model.eval()
    model.classifier.add_module('7', Softmax())

    explainer = WachterCounterfactual(model)

    fpath = 'data/pics/cat_1.jpg'

    orig_img = load_img(fpath, resize=(224, 224))
    img = load_img(fpath, imagenet=True)

    model_output = model(img)[0]
    most_probable_class = int(torch.argmax(model_output))
    print(f"Probability of {most_probable_class}[{clsidx_to_labels[most_probable_class]}] "
          f"class is {model_output[most_probable_class]}")

    counterfactual = explainer.generate(img, most_probable_class, to_maximize=False, threshold=0.1, grad_step=0.1, n_iters=10, verbose=True)

    new_model_output = model(counterfactual)[0]
    print(f"New probability of {most_probable_class} class is {new_model_output[most_probable_class]}")

    diff = counterfactual - img
    plot_row_images([img, counterfactual, diff],
                    titles=[
                        f"original image [P={round(model_output[most_probable_class].item(), 3)}]",
                        f"counterfactual [P={round(new_model_output[most_probable_class].item(), 3)}]",
                        f"difference between images"
                    ])
