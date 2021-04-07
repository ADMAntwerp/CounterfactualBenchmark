""" Process a MLPClassifier from Scikit-Learn

"""
from cfgenerator.model import functions


def get_layers(model):
    return model.coefs_, model.intercepts_


def get_functions(model, manual_act_functions):
    act_functions = []
    d_act_functions = []

    if manual_act_functions is None:
        # This calculates the number of layers excluding the output
        n_layers_not_output = len(model.coefs_) - 1

        function_name = model.get_params()["activation"]

        if function_name not in dir(functions):
            raise AttributeError(f"The activation function {function_name} is not supported.")

        act_functions.append(getattr(functions, function_name))
        d_act_functions.append(getattr(functions, "d_"+function_name))

        # As our optimizer for NN need a list with an activation function for each
        # single layer, we must multiply the act_functions and d_act_functions lists
        act_functions = act_functions*n_layers_not_output
        d_act_functions = d_act_functions*n_layers_not_output

        # For the MLPClassifier of Scikit-Learn, the output function is always logistic
        act_functions.append(functions.logistic)
        d_act_functions.append(functions.d_logistic)

    else:
        for act_fun_name in manual_act_functions:
            if act_fun_name not in dir(functions):
                raise AttributeError(f"The activation function {act_fun_name} is not supported.")

            act_functions.append(getattr(functions, act_fun_name))
            d_act_functions.append(getattr(functions, "d_" + act_fun_name))

    return act_functions, d_act_functions
