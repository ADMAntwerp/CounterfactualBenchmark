import os
import numpy as np
import pandas as pd
import json

columns = ['Age', 'Workclass', 'fnlgwt', 'Education', 'Education num', 'Marital Status',
           'Occupation', 'Relationship', 'Race', 'Sex', 'Capital Gain', 'Capital Loss',
           'Hours/Week', 'Native country', 'Income']
dir_path = os.path.dirname(os.path.realpath(__file__))
train = pd.read_csv(dir_path + '/' + 'adult-training.csv', names=columns)
test = pd.read_csv(dir_path + '/' + 'adult-test.csv', names=columns)
file_content = pd.concat([train, test], axis=0)
file_content['Age'] = file_content['Age'].astype(float)
file_content['fnlgwt'] = file_content['fnlgwt'].astype(float)
file_content['Education num'] = file_content['Education num'].astype(float)
file_content['Capital Gain'] = file_content['Capital Gain'].astype(float)
file_content['Capital Loss'] = file_content['Capital Loss'].astype(float)
file_content['Hours/Week'] = file_content['Hours/Week'].astype(float)

df = file_content

df['Income'] = df['Income'].apply(lambda x: 'High' if x == ' >50K' else 'Low')

for col in df.columns:
    if type(df[col][0]) == str:
        print("Working on " + col)
        df[col] = df[col].apply(lambda val: val.replace(" ", ""))

df.replace(' ?', np.nan, inplace=True)
df.dropna(inplace=True)

prefix = 0

to_remove = ['fnlgwt',  # Statistical weight for each instance. Assume irrelevant
             'Education']  # Nominal version of the Education column. Duplicate
print(df.columns)
df.drop(columns=to_remove, inplace=True)
print(df.columns)
nominals = ['Workclass', 'Marital Status',
           'Occupation', 'Relationship', 'Race', 'Sex', 'Native country', 'Income']
# process nominals

mapping = dict()
for prefix, col in enumerate(nominals):
    mapping[col] = dict()
    values = df[col].unique()
    for i, value in enumerate(values):
        label_num = int('%s%s' % (prefix + 1, i))
        mapping[col][value] = label_num
    df[col].replace(mapping[col], inplace=True)

for prefix, col in enumerate(nominals):
    values = df[col].unique()

idx = 0
features = list()
for i, col in enumerate(df.columns):
    feature = dict()
    if col in nominals:
        feature['type'] = 'nominal' if col != 'Income' else 'class'
        feature['name'] = col
        vals = list(mapping[col].values())
        feature['values'] = vals
        feature['num_values'] = len(vals)
        feature['idx'] = idx
        feature['i'] = i
        idx += len(vals)
    else:
        feature['type'] = 'numeric'
        feature['name'] = col
        feature['idx'] = idx
        feature['i'] = i
        feature['num_values'] = 1
        idx += 1
    features.append(feature)

data = np.array(df)

json.dump(mapping, open(dir_path + '/' + 'mapping.json', 'w'))
json.dump(features, open(dir_path + '/' + 'adult.features.json', 'w'))
np.savetxt(dir_path + '/' + 'adult.data', data, fmt='%d')