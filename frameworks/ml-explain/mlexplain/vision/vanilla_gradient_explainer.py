import torch
from torch.autograd import Variable
import torchvision
import numpy as np


class VanillaGradientExplainer:
    def __init__(self, model):
        self.model = model

    def explain(self, X, target_class):
        """Computes gradient of the model output w.r.t input X

        Parameters
        ----------
        X : str
            input to the model
        target_class : [int, numpy.array[float]]
            If specified as int, computes gradient of output target class.

            If specified as numpy.array[float] takes dot product between output and target.
            Usually used as one-hot encoding.

        Returns
        -------
        torch.tensor
            gradient of the output w.r.t input X
        """

        if isinstance(X, np.ndarray):
            X = torch.tensor(X)

        x_var = Variable(X, requires_grad=True)
        model_output = self.model(x_var)
        self.model.zero_grad()

        if isinstance(target_class, int):
            one_hot_target = torch.zeros_like(model_output)

            # Here we assume model_output has shape (#samples, #classes)
            one_hot_target[:, target_class] = 1
            target_class = one_hot_target

        model_output.backward(target_class)

        return x_var.grad


def main():
    from mlexplain.vision.utils import plot_img, plot_img_heatmap, load_img
    from mlexplain.utils import read_imagenet_clsidx_to_labels
    import torch.nn.functional as F

    clsidx_to_labels = read_imagenet_clsidx_to_labels('data/imagenet_1000_clsidx_to_labels.json')

    model = torchvision.models.vgg16(pretrained=True)
    model.eval()

    explainer = VanillaGradientExplainer(model)

    fpath = 'data/pics/cat_1.jpg'

    orig_img = load_img(fpath, resize=(224, 224))
    img = load_img(fpath, imagenet=True)

    model_output = F.softmax(model(img)[0], dim=0)
    most_probable_class = int(torch.argmax(model_output))
    print(f"Probability of {most_probable_class}[{clsidx_to_labels[most_probable_class]}] "
          f"class is {model_output[most_probable_class]}")

    explaining_result = explainer.explain(img, most_probable_class)

    new_model_output = F.softmax(model(img - 1.0 * explaining_result)[0], dim=0)
    print(f"New probability of {most_probable_class} class is {new_model_output[most_probable_class]}")

    plot_img_heatmap(orig_img, explaining_result, alpha=0.5)
    plot_img(orig_img - 1.0*explaining_result)


if __name__ == "__main__":
    main()
