""" Has all supported functions and their respective derivatives

"""
import numpy as np


def relu(array):
    return np.maximum(0, array)


def d_relu(array):
    return np.maximum(0, array)


def logistic(array):
    f = lambda x: 1 / (1 + np.exp(-x))
    return f(array)


def d_logistic(array):
    f = lambda x: np.exp(-x) / ((np.exp(-x) + 1) ** 2)
    return f(array)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def d_softmax(array):
    f = lambda x: np.exp(-x) / ((np.exp(-x) + 1) ** 2)
    return f(array)


def linear(x):
    return x


def d_linear(x):
    return 1


