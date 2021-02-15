import argparse

import time

from benchmark_template_row_instance import run_experiment

import numpy as np

import sys
sys.path.append('../frameworks/highgarden/highgarden/exploration')

import path_to_ennemies_featsel

framework_name = 'GROWINGSPHERES4'
output_number = 1  # The number of neural network outputs


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
    def pred(obs):
        if len(obs.shape) > 1 and obs.shape[1] > 1:
            return np.array([x[0] for x in adapted_nn.predict(obs)])
        else:
            return adapted_nn.predict(obs.reshape(1, -1))[0][0]

    def interpretability_metrics(X, prediction_function, interpretability_method, obs_to_interprete,
                                 **kwargs):
        nearest_ennemy = interpretability_method(X, prediction_function, obs_to_interprete, **kwargs)
        return nearest_ennemy

    # After the counterfactual generator setup and before the counterfactual generation, start a timer to measure the
    # time to generate it.
    start_time = time.time()

    # Here you generate the counterfactual
    cf_generation_output = interpretability_metrics(df_oh_train.drop(columns=['output']).to_numpy(),
                                                    pred,
                                                    path_to_ennemies_featsel.main,
                                                    np.array(factual_oh))

    # Just after the counterfactual generation, measure the time used to generate it
    time_cf_generation = time.time() - start_time

    # After that, you can adapt the counterfactual generation result to be in the required format (a simple list)
    if (len(cf_generation_output) > 0):
        cf = cf_generation_output.tolist()
    else:
        cf = []

    # Then, return the generated counterfactual (in the right format) and the time taken to generate it
    return cf, time_cf_generation


parser = argparse.ArgumentParser()
parser.add_argument("d")
parser.add_argument("c")
parser.add_argument("i")
args = parser.parse_args()

if __name__ == "__main__":
    if args.d:
        run_experiment(framework_name, framework_tester, int(args.c), int(args.i), output_number, [int(args.d)])
    else:
        run_experiment(framework_name, framework_tester, int(args.c), int(args.i), output_number)
