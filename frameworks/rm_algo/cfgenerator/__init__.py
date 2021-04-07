""" This main module will get the processed data, processed model and optimizer and
it will do the optimization.

"""
import sys
sys.path.append('./')

from cfgenerator.base.data_formatter import BaseForData
from cfgenerator.base.model_formatter import BaseForModel
from cfgenerator.optimizers.loss_functions import basic_loss

from cfgenerator.optimizers import nn_optimizer


def generate(x, cf_y, model, lr, steps, loss, cat_feats=None, act_functions=None, **kwargs):

    # Process data to have a pandas df always
    data_object = BaseForData(x, cf_y, cat_feats)

    # Process Model and get the needed parameters
    model_object = BaseForModel(model, act_functions).model_processed

    # Verify if loss function is supported
    if loss not in dir(basic_loss):
        raise AttributeError(f"The loss {loss} is not supported.")
    else:
        loss_func_dict = {"loss": getattr(basic_loss, loss), "d_loss": getattr(basic_loss, "d_"+loss)}

    if model_object["optimization_type"] == "neural_network":

        # Get the processed data
        x = data_object.formatted_one_hot_x
        cf_y = data_object.formatted_cf_y

        # Get the model parameters
        weights = model_object["weights"]
        biases = model_object["biases"]
        act_func = model_object["act_func"]
        d_act_func = model_object["d_act_func"]

        return nn_optimizer.optimize(x, cf_y, weights, biases, act_func,
                                     d_act_func, loss_func_dict, lr, steps, **kwargs)

