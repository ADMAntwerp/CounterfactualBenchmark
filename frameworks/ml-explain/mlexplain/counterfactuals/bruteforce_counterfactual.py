import torch
import numpy as np
from itertools import combinations
import torchvision
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn import Softmax


class BruteforceCounterfactual:
    def __init__(self, model, fill_func='mean'):
        self.model = model
        self.fill_func = self._get_fill_func(fill_func)

    # TODO (don't do optimization by yourself, use existing optimizer for it!)
    def generate(self, image, target_class, n_parts, K=10, fill_func='mean', verbose=False):
        smallest_prob = 1e10
        best_counterfactual = None

        for combination in combinations(range(K * K), n_parts):
            mask = np.ones(K * K, dtype=np.float32)
            mask[np.array(combination)] = 0.0
            mask = mask.reshape((K, K))
            mask = torch.from_numpy(mask)
            upsampled_mask = F.interpolate(mask.expand(1, 3, -1, -1), size=224)
            counterfactual = upsampled_mask * image
            cur_prob = self.model(counterfactual)[0, target_class]

            if cur_prob < smallest_prob:
                print(cur_prob)
                smallest_prob, best_counterfactual = cur_prob, counterfactual

        return smallest_prob, best_counterfactual

    @staticmethod
    def _get_fill_func(fill_func):
        if callable(fill_func):
            return fill_func

        name_to_func = {
            'mean': torch.dist
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
    def _min_fill_func(image, top_left_coord, bottom_right_coord):
        pass


if __name__ == "__main__":
    from mlexplain.vision.utils import plot_img, plot_img_heatmap, load_img, plot_row_images
    from mlexplain.utils import read_imagenet_clsidx_to_labels
    import torch.nn.functional as F

    clsidx_to_labels = read_imagenet_clsidx_to_labels('data/imagenet_1000_clsidx_to_labels.json')

    model = torchvision.models.vgg16(pretrained=True)
    model.eval()
    model.classifier.add_module('7', Softmax())

    explainer = BruteforceCounterfactual(model)

    fpath = 'data/pics/cat_1.jpg'

    orig_img = load_img(fpath, resize=(224, 224))
    img = load_img(fpath, imagenet=True)

    model_output = model(img)[0]
    most_probable_class = int(torch.argmax(model_output))
    print(f"Probability of {most_probable_class}[{clsidx_to_labels[most_probable_class]}] "
          f"class is {model_output[most_probable_class]}")

    smallest_prob, counterfactual = explainer.generate(img, most_probable_class, 1, verbose=True)
    print(smallest_prob)

    new_model_output = model(counterfactual)[0]
    print(f"New probability of {most_probable_class} class is {new_model_output[most_probable_class]}")

    diff = counterfactual - img
    plot_row_images([img, counterfactual, diff],
                    titles=[
                        f"original image [P={round(model_output[most_probable_class].item(), 3)}]",
                        f"counterfactual [P={round(new_model_output[most_probable_class].item(), 3)}]",
                        f"difference between images"
                    ])
