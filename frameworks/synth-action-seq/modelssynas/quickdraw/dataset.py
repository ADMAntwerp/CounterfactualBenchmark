import csv
import json
import numpy as np

from common.paths import QUICKDRAW_DIR


class QuickdrawDataset(object):
    def __init__(self, filename):
        dataset = np.load(filename)
        self.data = dataset['data']
        self.labels = dataset['labels']
        print(self.data[:32,0,:])
        print(self.data[32:64,0,:])


if __name__ == '__main__':
    dataset = np.load(str(QUICKDRAW_DIR / 'gold_cat.npz'))
    train = dataset['train']
    val = dataset['val']
    test = dataset['test']