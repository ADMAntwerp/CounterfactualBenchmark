import pandas as pd

from cfgenerator.data.labeled_x_cf import format_data
from cfgenerator.data.dataset import format_dataset


class ValidationErrors:
    @staticmethod
    def type_error(data, variable_name):
        if type(data) not in [pd.Series]:
            raise AttributeError(f"The {variable_name} is {str(type(data))} and not a Pandas "
                                 f"Series ( maybe try pandas.Series(<YOUR_DATA>) )")

    @staticmethod
    def single_x(data):
        if data.shape[0] != 1:
            raise AttributeError("This algorithm only supports single (1) original point feature"
                                 "(x) value")

    @staticmethod
    def check_range_index(data, variable_name):
        if list(data.index) != [*range(data.shape[0])]:
            raise AttributeError(f"Check the index of {variable_name}, it must be sequential "
                                 f"and start with zero (e.g. 0, 1,2,3,4...)")

    @staticmethod
    def verify_dataset_classes(y, cf_y, dataset, label_name):
        points_labels = list(y.unique()) + list(cf_y.unique())
        dataset_labels = list(dataset[label_name].unique())
        additional_point_labels = set.difference(set(points_labels + dataset_labels),
                                                 set(dataset_labels))
        if additional_point_labels:
            raise AttributeError(f"Your original point or cf point has a label "
                                 f"{additional_point_labels} that the dataset does not have")


class BaseForData:

    def __init__(self, x, cf_y, cat_feats=None):

        # Checks on data, They must be Series or DataFrames
        ValidationErrors.type_error(x, "original point feature (x)")
        if cf_y is not None:
            ValidationErrors.type_error(cf_y, "counterfactual point feature label (cf_y)")

        # Basic formatting to Pandas DF without one-hot encoding on categorical features
        self.formatted_x, self.formatted_cf_y = format_data(x, cf_y, cat_feats)

        # Verify if x has a single point
        ValidationErrors.single_x(self.formatted_x)
        # Check if ranges start with zero and are continuous
        ValidationErrors.check_range_index(self.formatted_x,
                                           "original point feature (x)")
        ValidationErrors.check_range_index(self.formatted_cf_y,
                                           "counterfactual point feature label (cf_y)")

        # Formatting including one-hot encoding for categorical features
        if cat_feats:
            self.formatted_one_hot_x, _ = format_data(x, cf_y, cat_feats, True)

            self.cat_feats = cat_feats
        else:
            # If there is no categorical features, the one_hot representation will be the same
            # as the normal representation
            self.formatted_one_hot_x = self.formatted_x

            self.cat_feats = []