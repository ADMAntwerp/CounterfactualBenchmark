import os
import numpy as np


class FannieMaeDataset(object):
    def __init__(self, filename=os.path.join('models','fanniemae','fanniemae_z_full.test.npy')):
        dataset = np.load(filename)
        self.data = dataset[:, :-2]
        self.labels = dataset[:,-2:]

