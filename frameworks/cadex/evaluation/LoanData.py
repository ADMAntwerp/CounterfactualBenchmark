import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

class LoanData:
    '''
    Encapsulates the dataset. Loads data from files, splits into training/validation, and normalizes the fields
    '''
    def __init__(self, input_file, labels_file, attributes_file, valid_frac=0.2, test_frac=0):
        '''
        Initialize the dataset from given files
        :param input: CSV file with input data
        :param labels: CSV file with data labels
        :param category_map: map from categorical attribute to list of one-hot encoded attributes
        :param nrows: number of rows to load from the file
        :param test_frac: percentage of data to split into test set
        :param valid_frac: percentage of data to split into validation set
        :param filter_columns: specific columns to not load from the file
        :param scale: set to True in order to normalize the attributes
        '''
        self.input = pd.read_csv(input_file, index_col=0)
        self.labels = pd.read_csv(labels_file, index_col=0, dtype=np.int32)
        self.index = 0

        # Split into training and validation sets
        non_test_input, self.test_input, non_test_labels, self.test_labels = \
            train_test_split(self.input, self.labels, test_size=test_frac)
        self.train_input, self.valid_input, self.train_labels, self.valid_labels = \
            train_test_split(non_test_input, non_test_labels, test_size=valid_frac)

        self.num_input_columns = len(self.input.columns)
        self.input_columns = self.input.columns
        self.num_label_columns = len(self.labels.columns)
        self.label_columns = self.labels.columns

        attributes = pickle.load(open(attributes_file, 'rb'))
        # Load category map
        self.categorical_attributes = attributes['categorical']
        self.ordinal_attributes = attributes['ordinal']

        category_col_names = [j for i in self.categorical_attributes for j in i]
        self.cat_indices = [col in category_col_names for col in self.input.columns]

        # Normalize to mean 0, std 1. Only scale non-categorical columns
        self._scaler = StandardScaler()
        self._scaler.fit(self.train_input)
        self._scaler.scale_[self.cat_indices] = 1
        self._scaler.mean_[self.cat_indices] = 0
        self.train_input = pd.DataFrame(self._scaler.transform(self.train_input), index=self.train_input.index,
            columns=self.train_input.columns)
        if self.valid_input.shape[0] > 0:
            self.valid_input = pd.DataFrame(self._scaler.transform(self.valid_input), index=self.valid_input.index,
                columns=self.valid_input.columns)

    def unscale(self, data):
        '''Unscale an input vector to return it to original input range'''
        if type(data) is pd.DataFrame:
            return pd.DataFrame(self._scaler.inverse_transform(data), index=data.index, columns=data.columns)
        elif type(data) is pd.Series:
            return pd.Series(self._scaler.inverse_transform(data), index=data.index)
        else:
            return self._scaler.inverse_transform(data)

    def scale(self, data):
        '''Unscale an input vector to return it to original input range'''
        if type(data) is pd.DataFrame:
            return pd.DataFrame(self._scaler.transform(data), index=data.index, columns=data.columns)
        elif type(data) is pd.Series:
            return pd.Series(self._scaler.transform(data), index=data.index)
        else:
            return self._scaler.transform(data)

class IdentityScaler:
    '''Dummy class which doesn't do actual scaling'''
    def inverse_transform(self, x):
        return x

