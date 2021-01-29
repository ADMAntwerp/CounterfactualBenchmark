import argparse

import time

from benchmark_template import run_experiment

import numpy as np

import sys
sys.path.append('../frameworks/mace/')

import pandas as pd
from sklearn.neural_network import MLPClassifier

import generateSATExplanations
from loadData import loadDataset

framework_name = 'MACE'
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
    bin_feats = converter.binary_cats if cat_feats else []

    df_input = pd.concat([df_train, pd.DataFrame([[*factual, 0]], columns=list(df_train.columns))])

    if cat_feats:
        for c_feat in list(set(cat_feats) - set(bin_feats)):
            dict_conversion = {}
            for idx_cat_key, key_ordered_n in enumerate(converter.dict_feat_idx[c_feat].keys()):
                if df_input[c_feat].dtype == np.int64:
                    dict_conversion[int(float(key_ordered_n))] = idx_cat_key + 1
                else:
                    dict_conversion[key_ordered_n] = idx_cat_key + 1
            df_input[c_feat] = df_input[c_feat].map(dict_conversion)

    df_input = pd.concat([df_input[['output']], df_input.drop(columns=['output'])], axis=1)
    ds = loadDataset('generic', return_one_hot=False if not cat_feats else True, df=df_input, df_oh=None,
                     cat_feats=cat_feats, num_feats=num_feats, bin_feats=bin_feats)
    ds.data_frame_kurz = ds.data_frame_kurz.iloc[:-1]
    ds.data_frame_long = ds.data_frame_long.iloc[:-1]

    cf_ds_kurz = ds.data_frame_kurz.iloc[-1:].copy()

    layerStructure = []
    for layer_idx, layer in enumerate(adapted_nn.layers):
        if layer_idx != 0:
            layerStructure.append(layer.get_weights()[0].shape[0])

    model_skl = MLPClassifier(layerStructure[1:], max_iter=1)

    model_skl.fit(ds.data_frame_kurz.drop(columns=['y']).to_numpy(), ds.data_frame_kurz[['y']].to_numpy())

    for layer_idx, layer in enumerate(adapted_nn.layers):
        if layer_idx != 0:
            model_skl.coefs_[layer_idx-1] = layer.get_weights()[0]
            model_skl.intercepts_[layer_idx-1] = layer.get_weights()[1]

    factual_sample = cf_ds_kurz.iloc[0].to_dict()
    factual_sample['y'] = True if factual_sample['y'] else False

    # After the counterfactual generator setup and before the counterfactual generation, start a timer to measure the
    # time to generate it.
    start_time = time.time()

    # Here you generate the counterfactual
    cf_generation_output = generateSATExplanations.genExp(
                        'test',
                        model_skl,
                        ds,
                        factual_sample,
                        'zero_norm',
                        'mace',
                        1e-5
                    )

    # Just after the counterfactual generation, measure the time used to generate it
    time_cf_generation = time.time() - start_time

    # After that, you can adapt the counterfactual generation result to be in the required format (a simple list)
    if cf_generation_output is not None:
        cf = pd.DataFrame([cf_generation_output['cfe_sample']]).drop(columns=['y']).to_numpy().tolist()[0]
    else:
        cf = []

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
