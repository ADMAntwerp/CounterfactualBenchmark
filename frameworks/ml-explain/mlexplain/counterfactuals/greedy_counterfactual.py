import torch
import numpy as np
from itertools import combinations
import torchvision
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn import Softmax


class GreedyCounterfactual:
    def __init__(self, model, fill_func='mean'):
        self.model = model
        self.fill_func = self._get_fill_func(fill_func)

    # TODO (don't do optimization by yourself, use existing optimizer for it!)
    def generate(self, image, target_class, max_n_parts=2, K=10, verbose=False):
        def generate_counterfactual(img, mask):
            mask = mask.reshape((K, K))
            counterfactual = self.fill_func(img, mask)
            cur_prob = self.model(counterfactual)[0, target_class]
            return cur_prob, counterfactual

        def find_next_best_counterfactual(img, mask):
            smallest_prob = 1e10
            best_counterfactual = None
            best_mask = None

            for idx in range(K * K):
                if mask[idx] == 1:
                    continue
                new_mask = mask.copy()
                new_mask[idx] = 1.0
                cur_prob, counterfactual = generate_counterfactual(img, new_mask)

                if cur_prob < smallest_prob:
                    smallest_prob, best_counterfactual, best_mask = cur_prob, counterfactual, new_mask

            return smallest_prob, best_counterfactual, best_mask

        counterfactuals = []
        probabilities = []

        counterfactuals.append(image)
        probabilities.append(self.model(image)[0, target_class])

        best_mask = np.zeros(K * K, dtype=np.float32)
        for idx in range(max_n_parts):
            smallest_prob, best_counterfactual, best_mask = find_next_best_counterfactual(counterfactuals[-1], best_mask)
            counterfactuals.append(best_counterfactual)
            probabilities.append(smallest_prob)

        return probabilities, counterfactuals

    @staticmethod
    def _get_fill_func(fill_func):
        if callable(fill_func):
            return fill_func

        # If there is one, we apply mask, otherwise we don't.
        name_to_func = {
            'gray': GreedyCounterfactual._gray_fill_func,
            'mean': GreedyCounterfactual._mean_fill_func,
            'noise': GreedyCounterfactual._noise_fill_func
        }

        if isinstance(fill_func, str):
            if fill_func not in name_to_func:
                raise RuntimeError("Fill function '{}' not found. Provide your own or use own of the following "
                                   "types: {}".format(fill_func, list(name_to_func.keys())))

            return name_to_func[fill_func]
        else:
            raise RuntimeError("Type {} for fill_func is not supported. Either use name of the function, "
                               "or provide your own function".format(type(fill_func)))

    @staticmethod
    def _gray_fill_func(image, mask):
        mask = torch.from_numpy(mask)
        upsampled_mask = F.interpolate(mask.expand(1, 3, -1, -1), size=image.shape[-2])
        counterfactual = (1 - upsampled_mask) * image
        return counterfactual

    @staticmethod
    def _mean_fill_func(image, mask):
        mask = torch.from_numpy(mask)
        upsampled_mask = F.interpolate(mask.expand(1, 3, -1, -1), size=image.shape[-2])
        mean_value = torch.sum(upsampled_mask * image) / (torch.sum(upsampled_mask > 0))
        counterfactual = (1 - upsampled_mask) * image + upsampled_mask * mean_value
        return counterfactual

    @staticmethod
    def _noise_fill_func(image, mask):
        mask = torch.from_numpy(mask)
        upsampled_mask = F.interpolate(mask.expand(1, 3, -1, -1), size=image.shape[-2])
        counterfactual = (1 - upsampled_mask) * image + upsampled_mask * torch.clamp(torch.randn_like(upsampled_mask), -1.0, 1.0)
        return counterfactual


if __name__ == "__main__":
    from mlexplain.vision.utils import plot_img, plot_img_heatmap, load_img, plot_grid_images
    from mlexplain.utils import read_imagenet_clsidx_to_labels
    import torch.nn.functional as F

    clsidx_to_labels = read_imagenet_clsidx_to_labels('data/imagenet_1000_clsidx_to_labels.json')

    model = torchvision.models.vgg16(pretrained=True)
    model.eval()
    model.classifier.add_module('7', Softmax())

    fill_func = 'noise'
    explainer = GreedyCounterfactual(model, fill_func=fill_func)

    for fpath in ['data/pics/cat_1.jpg', 'data/pics/cat_2.jpg', 'data/pics/cat_3.jpg',
                  'data/pics/cat_4.jpg', 'data/pics/cat_5.jpg', 'data/pics/cat_6.jpg']:

        # fpath = 'data/pics/cat_2.jpg'
        orig_img = load_img(fpath, resize=(224, 224))
        img = load_img(fpath, imagenet=True)

        model_output = model(img)[0]
        most_probable_class = int(torch.argmax(model_output))
        # tmp to have the same results
        # most_probable_class = 285
        print(f"Probability of {most_probable_class}[{clsidx_to_labels[most_probable_class]}] "
              f"class is {model_output[most_probable_class]}")

        n_counterfactuals = 8
        probs, counterfactuals = explainer.generate(img, most_probable_class, n_counterfactuals, verbose=True)

        titles = [f"counterfactual-{idx} [P={round(prob.item(), 3)}]" for idx, prob in enumerate(probs)]
        nrows = ncols = int(np.sqrt(n_counterfactuals)) + 1
        plot_grid_images(counterfactuals, nrows, ncols, titles=titles,
                         save=True, fname=f'{fpath.split("/")[-1].split(".")[0]}_greedy_{fill_func}_{n_counterfactuals}.jpg')
