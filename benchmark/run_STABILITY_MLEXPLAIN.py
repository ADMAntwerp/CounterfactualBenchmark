import argparse

import time

from benchmark_template import run_experiment

import torch
import torch.nn as nn

import sys
sys.path.append('../frameworks/ml-explain/')

from mlexplain.counterfactuals.vanilla_gradient_counterfactual import VanillaGradientCounterfactual

framework_name = 'MLEXPLAIN'
output_number = 2  # The number of neural network outputs


def framework_tester(df_train, df_oh_train, df_test, df_oh_test, num_feats, cat_feats, converter, adapted_nn,
                     df_factual, factual, factual_oh, session):
    """ Function used to benchmark counterfactual explanation generation algorithms. It includes most data one generator
    may use, although it's not needed to use all them. Please, report if you think any additional data should be
    provided. This function output must be a simple list with the counterfactual result and the time used to generate
    it.

    :param df_train: The train dataframe, including the output class
    :param df_oh_train: Same as df_train but one-hot encoded, IF THERE'S NO CATEGORICAL FEATURES IT'S THE SAME AS df_train
    :param df_test: The test dataframe, including the output class
    :param df_oh_test: Same as df_test but one-hot encoded, IF THERE'S NO CATEGORICAL FEATURES IT'S THE SAME AS df_test
    :param num_feats: A list with the column names of numerical features
    :param cat_feats: A list with the column names of categorical features
    :param converter: Converter from/to one-hot encoded.
                        Conversion non-encoded -> one-hot encoded: converter.convert_to_oh(INPUT_DATA_NON_ENCODED).
                        Conversion one-hot encoded -> non-encoded: converter.convert(INPUT_DATA_OH_DATA).
                        * INPUT_DATA_NON_ENCODED and INPUT_DATA_OH_DATA must be simple lists
                        * INPUT_DATA_NON_ENCODED and INPUT_DATA_OH_DATA must follow the same column structure as df_train or df_oh_train
                        -> PROPERTY: converter.binary_cats - Binary's features column names
    :param adapted_nn: TensorFlow/Keras neural network.
                        Neural network weights can be accessed by using: adapted_nn.get_weights(), for more info
                        please, refer to TensorFlow/Keras documentation
    :param df_factual: Dataset with the factual rows to be tested
    :param factual: A simple list of the factual result to be tested
    :param factual_oh: Same as factual but one-hot encoded, IF THERE'S NO CATEGORICAL FEATURES IT'S THE SAME AS factual
    :param session: TensorFlow current session

    :return: 2 outputs:
        (list) - A simple list with the counterfactual result. It can be one-hot encoded or not.
                 If a counterfactual is not found, return a empty list.
        (float) - Time used to generate a counterfactual
    """

    # First you must configure your generator, this may be fit with training data or process neural network parameters
    # or even convert data or the model. These procedures are those needed before the cf generation and generally just
    # done one time.
    class SimpleNN(nn.Module):
        def __init__(self):
            super().__init__()

            shapeL1 = adapted_nn.layers[1].get_weights()[0].shape
            self.layer1 = nn.Linear(shapeL1[0], shapeL1[1])

            shapeLO = adapted_nn.layers[2].get_weights()[0].shape
            self.layerO = nn.Linear(shapeLO[0], shapeLO[1])

        def forward(self, x):
            """ Accepts input of shape (N_objects, n_input) """
            x = self.layer1(x)
            x = torch.relu(x)
            x = self.layerO(x)
            x = torch.softmax(x, 1)
            return x

    classifier = SimpleNN()

    classifier.layer1.weight.data = torch.tensor(adapted_nn.layers[1].get_weights()[0].T)
    classifier.layer1.bias.data = torch.tensor(adapted_nn.layers[1].get_weights()[1])

    classifier.layerO.weight.data = torch.tensor(adapted_nn.layers[2].get_weights()[0].T)
    classifier.layerO.bias.data = torch.tensor(adapted_nn.layers[2].get_weights()[1])

    cfg = VanillaGradientCounterfactual(classifier)

    # After the counterfactual generator setup and before the counterfactual generation, start a timer to measure the
    # time to generate it.
    start_time = time.time()

    # Here you generate the counterfactual
    cf_generation_output = cfg.generate(torch.tensor([factual_oh], dtype=torch.float32), [[1]])

    # Just after the counterfactual generation, measure the time used to generate it
    time_cf_generation = time.time() - start_time

    # After that, you can adapt the counterfactual generation result to be in the required format (a simple list)
    cf = cf_generation_output.detach().numpy().tolist()[0]

    # Then, return the generated counterfactual (in the right format) and the time taken to generate it
    return cf, time_cf_generation


parser = argparse.ArgumentParser()
parser.add_argument("d")
args = parser.parse_args()

if __name__ == "__main__":
    if args.d:
        run_experiment(framework_name, framework_tester, output_number, [int(args.d)])
    else:
        run_experiment(framework_name, framework_tester, output_number)
