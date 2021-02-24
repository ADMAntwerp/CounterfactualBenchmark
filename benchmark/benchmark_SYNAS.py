import argparse

import time

from benchmark_template_row_instance import run_experiment

import sys
sys.path.append('../frameworks/synth-action-seq/')

import numpy as np

from modelssynas import loader
from heuristics.loader import load_heuristics
from recourse.search import SequenceSearch
from recourse.config import base_config

framework_name = 'SYNAS'
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
    class model_synas:
        def __init__(self, adapted_nn):
            self.FALSE_LABEL = [0.0, 1.0]
            self.TRUE_LABEL = [1.0, 0.0]
            self.input_dim = adapted_nn.layers[1].get_weights()[0].shape[0]
            self.model = adapted_nn

    model_synas_nn = model_synas(adapted_nn)

    bin_feats = converter.binary_cats if cat_feats else []
    dict_feat_idx = converter.dict_feat_idx if cat_feats else []

    data, actions, features, target_label = loader.setup_generic(df_oh_train, cat_feats, num_feats, bin_feats,
                                                                 dict_feat_idx)

    for name, feature in features.items():
        feature.initialize_tf_variables()

    heuristics = load_heuristics('vanilla', actions, model_synas_nn, 1)
    cfg = SequenceSearch(model_synas_nn, actions, heuristics, sav_dir=None, config=base_config)

    # After the counterfactual generator setup and before the counterfactual generation, start a timer to measure the
    # time to generate it.
    start_time = time.time()

    # Here you generate the counterfactual
    cf_generation_output = cfg.find_correction(np.array(factual_oh).reshape((1, len(factual_oh))), np.array([target_label]), session)

    # Just after the counterfactual generation, measure the time used to generate it
    time_cf_generation = time.time() - start_time

    # After that, you can adapt the counterfactual generation result to be in the required format (a simple list)
    if cf_generation_output.best_result is not None:
        cf = cf_generation_output.best_result.final_instance.tolist()
    else:
        cf = []

    # Then, return the generated counterfactual (in the right format) and the time taken to generate it
    return cf, time_cf_generation


parser = argparse.ArgumentParser()
parser.add_argument("d")
parser.add_argument("c")
parser.add_argument("i")
parser.add_argument("f")
args = parser.parse_args()

if __name__ == "__main__":
    if args.d:
        run_experiment(framework_name, framework_tester, int(args.c), int(args.i), output_number, [int(args.d)], str(args.f))
    else:
        run_experiment(framework_name, framework_tester, int(args.c), int(args.i), output_number)