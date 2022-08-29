from cfbench.benchmark_template_row_instance import BenchmarkGenerator


class BenchmarkCF:

    def __init__(self, framework_name: str, output_number: int = 1, disable_gpu: bool = False):
        self.framework_name = framework_name
        self.output_number = output_number
        self.disable_gpu = disable_gpu

    def framework_wrapper(
            self,
            df_train,
            df_oh_train,
            df_test,
            df_oh_test,
            num_feats,
            cat_feats,
            converter,
            adapted_nn,
            df_factual,
            factual,
            factual_oh,
            session):
        """
        Function used to benchmark counterfactual explanation generation algorithms. It includes most data one
        generator may use, although it's not needed to use all them. Please, report if you think any additional
        data should be provided. This function output must be a simple list with the counterfactual result and the
        time used to generate
        it.

        :param df_train: The train dataframe, including the output class
        :param df_oh_train: Same as df_train but one-hot encoded, IF THERE'S NO CATEGORICAL
        FEATURES IT'S THE SAME AS df_train
        :param df_test: The test dataframe, including the output class
        :param df_oh_test: Same as df_test but one-hot encoded, IF THERE'S NO CATEGORICAL FEATURES
        IT'S THE SAME AS df_test
        :param num_feats: A list with the column names of numerical features
        :param cat_feats: A list with the column names of categorical features
        :param converter: Converter from/to one-hot encoded.
                            Conversion non-encoded -> one-hot encoded: converter.convert_to_oh(INPUT_DATA_NON_ENCODED).
                            Conversion one-hot encoded -> non-encoded: converter.convert(INPUT_DATA_OH_DATA).
                            * INPUT_DATA_NON_ENCODED and INPUT_DATA_OH_DATA must be simple lists
                            * INPUT_DATA_NON_ENCODED and INPUT_DATA_OH_DATA must follow the same column structure as
                            df_train or df_oh_train
                            -> PROPERTY: converter.binary_cats - Binary's features column names
        :param adapted_nn: TensorFlow/Keras neural network.
                            Neural network weights can be accessed by using: adapted_nn.get_weights(), for more info
                            please, refer to TensorFlow/Keras documentation
        :param df_factual: Dataset with the factual rows to be tested
        :param factual: A simple list of the factual result to be tested
        :param factual_oh: Same as factual but one-hot encoded, IF THERE'S NO CATEGORICAL FEATURES IT'S THE SAME
        AS factual
        :param session: TensorFlow current session

        :return: 2 outputs:
            (list) - A simple list with the counterfactual result. It can be one-hot encoded or not.
                     If a counterfactual is not found, return an empty list.
            (float) - Time used to generate a counterfactual
        """
        raise (NotImplementedError, 'Please, implement this method')

    def create_generator(
            self,
            dataset_idx: list = None):
        # Verify if indexes of dataset_idx are in the valid range (0 - 21)
        if dataset_idx is not None:
            assert all([0 <= idx <= 21 for idx in dataset_idx]), 'Invalid dataset index, must be in range [0, 21]'
        else:
            dataset_idx = [*range(22)]

        return BenchmarkGenerator(
            framework_name=self.framework_name,
            output_number=self.output_number,
            ds_id_test=dataset_idx,
            disable_gpu=self.disable_gpu)
