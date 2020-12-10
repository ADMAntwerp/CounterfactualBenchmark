import torch
from mlexplain.sce.abstract_sce import AbstractSCE
import matplotlib.pyplot as plt
from torch import nn

from mlexplain.vision.utils import plot_img_heatmap


class ImageSCE(AbstractSCE):
    def __init__(self, model, target_class):
        super(ImageSCE, self).__init__(model, target_class)

    def init_masks(self, X):
        # We assume that X has the format (N, # of channels, height, width)
        return torch.zeros((X.shape[0], X.shape[1]), requires_grad=True)

    def perturbation_operator(self, X, masks):
        # We unsqueeze masks to match it shape of X, then clamp it to be positive
        return (X + masks.unsqueeze(2).unsqueeze(3)).clamp_(0.0, 1e9)

    def plot_feature_indices_heatmaps(self, images, feature_indices):
        fig = plt.figure(figsize=(len(feature_indices) * 4, self.X.shape[0] * 4))

        cur_idx = 1
        for img_idx in range(self.X.shape[0]):
            for idx, feature_idx in enumerate(feature_indices):
                heatmap = self.X[img_idx, feature_idx, :, :].unsqueeze(0).unsqueeze(0)
                heatmap = nn.Upsample(size=images.shape[2:], mode='bilinear')(heatmap)

                ax = fig.add_subplot(self.X.shape[0], len(feature_indices), cur_idx)
                cur_idx += 1
                if img_idx == 0:
                    ax.set_title(f'Feature index: {feature_idx}', fontsize=16)

                plot_img_heatmap(images[img_idx], heatmap, ax=ax, alpha=0.5)

        return fig

    def plot_mask_weighted_heatmaps(self, images):
        fig = plt.figure(figsize=(8, len(images) * 4))

        cur_idx = 1
        for img_idx in range(images.shape[0]):
            heatmap = 0
            for feature_idx, mask_entry in enumerate(self.masks[img_idx]):
                if mask_entry.item() < 0:
                    heatmap -= mask_entry * self.X[img_idx, feature_idx, :, :].unsqueeze(0).unsqueeze(0)

            heatmap = nn.Upsample(size=images.shape[2:], mode='bilinear')(heatmap)

            ax = fig.add_subplot(images.shape[0], 1, cur_idx)
            cur_idx += 1

            plot_img_heatmap(images[img_idx], heatmap, ax=ax, alpha=0.5)
        return fig
