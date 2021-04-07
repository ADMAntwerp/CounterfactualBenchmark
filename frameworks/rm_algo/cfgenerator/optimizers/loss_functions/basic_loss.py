import numpy as np


def rmse(y, cf_y, **kwargs):
    return (y - cf_y)**2


def d_rmse(y, cf_y, **kwargs):
    return 2*(y - cf_y)


def crossentropy(y, yHat, **kwargs):
    if y == 1:
      return -np.log(yHat)
    else:
      return -np.log(1 - yHat)


def d_crossentropy(y, yHat, **kwargs):
    if y == 1:
      return -1.0/yHat
    else:
      return -1.0/(1 - yHat)


def hinge(y, cf_y, **kwargs):

    if "hinge_higher" not in kwargs:
        kwargs["hinge_higher"] = True

    if kwargs["hinge_higher"]:
        return y - cf_y if y > cf_y else 0
    return y - cf_y if y < cf_y else 0


def d_hinge(y, cf_y, **kwargs):

    if "hinge_higher" not in kwargs:
        kwargs["hinge_higher"] = True

    if kwargs["hinge_higher"]:
        return y - cf_y if y > cf_y else 0
    return y - cf_y if y < cf_y else 0
