# LORE - LOcal Rule-based Explanations

This repository contains the source code of LORE (LOcal Rule-based Explanations) and the datasets used in the experiments presented in the paper "Local Rule-based Explanations of Black Box Decision Systems".


## How to run

`pip install -r requirements.txt`

Go to the `test_lore.py` file and modify the code below `if __name__ == "__main__":`, with the information of the dataset you want to retrieve counterfactuals. Then, run that file `python test_lore.py`.

## How to use custom model
The dataframe structure must be specific:
The columns names must be number starting from 0

For categorical, one-hot features, it must follow a structure `Number_featureName`, for example, for a feature (that must have a numerical name, like `4`) with values: `no_education`, `high_school`, `college` the one-hot columns MUST be like `4_no_education`, `4_high_school`, `4_college`.

The class `Converter` in the `test_lore.py` file handles the conversion from the LORE dataset format to a common one-hot format that can be processed by several models.

The class `ModelBlackBoxTF` in the `test_lore.py` file creates the model to be used, and it must have a converter to generate a data in a trainable format for most algorithms.: