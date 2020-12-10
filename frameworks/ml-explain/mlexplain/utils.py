import pandas as pd
import numpy as np
from PIL import Image
import json


def read_imagenet_clsidx_to_labels(fpath):
    with open(fpath) as f:
        clsidx_to_labels = json.loads(f.read())

    clsidx_to_labels = {int(idx): lbls[1] for idx, lbls in clsidx_to_labels.items()}

    return clsidx_to_labels
