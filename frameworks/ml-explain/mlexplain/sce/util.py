import torch


def choose_k_top_elements_flat(model, data, K=10, target_class=1):
    """
    Selects K elements from data, which have the biggest probability of target_class according to model.
    "_flat" means that model(data) outputs tensor of size [N, 1] with elements = probabilities of class 1.

    Parameters
    ----------
    model
    Pytorch model, subclass of nn.module

    data
    input to the model

    Returns
    -------
    Subarray of data with K elements, which have the biggest probabilities of class target_class
    """

    descending = True if target_class == 1 else False
    indices = torch.argsort(model(data).view(-1), dim=0, descending=descending)[:K]
    return data[indices, :]


def choose_k_top_elements_softmax(model, data, K=10, target_class=1):
    """
    Selects K elements from data, which have the biggest probability of target_class according to model.
    "_softmax" means that model(data) produces tensor of size (N, P), where P is the number of classes,
    and each element is a probability of corresponding class.

    Parameters
    ----------
    model
    Pytorch model, subclass of nn.module

    data
    input to the model

    Returns
    -------
    Subarray of data with K elements, which have the biggest probabilities of class target_class
    """

    indices = torch.argsort(model(data)[:, target_class].view(-1), dim=0, descending=True)[:K]
    return data[indices, :]
