from sklearn.neural_network import MLPClassifier

from cfgenerator.model import skl_MLPClassifier


class ValidationErrors:
    @staticmethod
    def model_type_error(model):
        if type(model) not in [MLPClassifier]:
            raise AttributeError(f"The model {type(model)} is not supported.")


class BaseForModel:

    def __init__(self, model, act_functions):

        # Check if model is supported by our algorithm
        ValidationErrors.model_type_error(model)

        self.model_processed = {}

        # This is the base pattern for a NN
        if type(model) == MLPClassifier:
            # MLPClassifier is a Scikit-Learn model that makes a simple Neural Network
            self.model_processed["optimization_type"] = "neural_network"

            # Get Layers weights and biases
            self.model_processed["weights"], \
            self.model_processed["biases"] = skl_MLPClassifier.get_layers(model)

            # Get activation functions and their derivative correspondents
            self.model_processed["act_func"], \
            self.model_processed["d_act_func"] = skl_MLPClassifier.get_functions(model, act_functions)





