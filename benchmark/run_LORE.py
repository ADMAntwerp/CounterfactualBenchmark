import argparse

import time

from benchmark_template import run_experiment

import pandas as pd

import sys
sys.path.append('../frameworks/LORE/')

from frameworks.LORE import test_lore

framework_name = 'LORE'
output_number = 1  # The number of neural network outputs


class Converter:
    def __init__(self, full_df, cat_feats, oh_col):
        self.col = list(full_df.columns)
        self.oh_col = oh_col
        self.oh_dict = {}
        self.cat_feats = cat_feats
        self.binary_cats = []

        for cat_f in cat_feats:
            if len(full_df[cat_f].unique()) <= 2:
                self.binary_cats.append(cat_f)

        for cat_f in cat_feats:
            values = list(full_df[cat_f].unique())
            for v in values:
                cat_feat_simple = f'{cat_f}_{v}'
                cat_feat_ordered = f'{"".join(["0"]*(4-len(str(cat_f))))}{str(cat_f)}_{v}'
                cat_feat_binary_ordered = f'{"".join(["0"]*(4-len(str(cat_f))))}{str(cat_f)}'

                if cat_feat_simple in oh_col:
                    self.oh_dict[cat_feat_simple] = {cat_f: v}
                elif cat_feat_ordered in oh_col:
                    self.oh_dict[cat_feat_ordered] = {cat_f: v}
                elif cat_feat_binary_ordered in oh_col:
                    self.oh_dict[cat_feat_binary_ordered] = {cat_f: 'binary'}
                else:
                    self.oh_dict[cat_f] = {cat_f: 'binary'}

        self.dict_feat_idx = {}
        for idx, col in enumerate(self.oh_col):
            cat_name = col.split('_')[0]
            value_name = '_'.join(col.split('_')[1:])
            if cat_name != 'output':
                if str(int(cat_name)) in self.dict_feat_idx.keys():
                    self.dict_feat_idx[str(int(cat_name))][value_name] = idx
                else:
                    self.dict_feat_idx[str(int(cat_name))] = {value_name: idx}

    def convert_to_oh(self, data):
        out_data = [0]*(len(self.oh_col) - 1)

        for idx, c_value in enumerate(data):
            if str(idx) in self.cat_feats and str(idx) not in self.binary_cats:
                try:
                    idx_oh = self.dict_feat_idx[str(idx)][c_value]
                except KeyError:
                    try:
                        idx_oh = self.dict_feat_idx[str(idx)][str(c_value)]
                    except KeyError:
                        idx_oh = self.dict_feat_idx[str(idx)][str(int(float(c_value)))]

                out_data[idx_oh] = 1
            else:
                try:
                    idx_oh = self.oh_col.index(str(idx))
                except ValueError:
                    len_idx = len(str(idx))
                    extended_idx = '0'*(4 - len_idx)+str(idx)
                    idx_oh = self.oh_col.index(extended_idx)

                out_data[idx_oh] = c_value

        return out_data

    def convert(self, data):
        if type(data) == pd.DataFrame:
            data = data.to_numpy().tolist()
        output = []
        for d in data:
            out_row = [0]*(len(self.col) - 1)
            for idx, f in enumerate(d):
                of_feat_col = self.oh_col[idx]
                if of_feat_col in self.oh_dict.keys():
                    col_idx, v = list(self.oh_dict[of_feat_col].items())[0]
                    if v == 'binary':
                        out_row[int(col_idx)] = f
                    else:
                        if f == 1:
                            out_row[int(col_idx)] = v
                else:
                    out_row[int(of_feat_col)] = f
            output.append(out_row)

        return output


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

    # converterLORE = Converter(df_train, cat_feats, list(df_oh_train.columns))

    # After the counterfactual generator setup and before the counterfactual generation, start a timer to measure the
    # time to generate it.
    start_time = time.time()

    # Here you generate the counterfactual
    cf_generation_output = test_lore.main(df_train, cat_feats, 'output', list(df_oh_train.columns), adapted_nn, pd.DataFrame([[*factual, 0]], columns=list(df_train.columns)))

    # Just after the counterfactual generation, measure the time used to generate it
    time_cf_generation = time.time() - start_time

    # After that, you can adapt the counterfactual generation result to be in the required format (a simple list)
    if len(cf_generation_output) > 0:
        cf = cf_generation_output.to_numpy()[0].tolist()
    else:
        cf = []

    # Then, return the generated counterfactual (in the right format) and the time taken to generate it
    return cf, time_cf_generation


run_experiment(framework_name, framework_tester, output_number, [10])
# parser = argparse.ArgumentParser()
# parser.add_argument("d")
# args = parser.parse_args()
#
# if __name__ == "__main__":
#     if args.d:
#         run_experiment(framework_name, framework_tester, output_number, [int(args.d)])
#     else:
#         run_experiment(framework_name, framework_tester, output_number)
