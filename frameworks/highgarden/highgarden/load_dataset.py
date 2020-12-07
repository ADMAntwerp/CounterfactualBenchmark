from sklearn.datasets import make_classification
import numpy as np
import pandas as pd
import random


DATASET_FILE = 'datasets/'

def generate_data(n_obs, n_features, n_informative, n_redundant, n_repeated, n_clusters_per_class):
    return make_classification(n_samples=n_obs, 
                           n_features=n_features, 
                           n_informative=n_informative, 
                           n_redundant=n_redundant, 
                           n_repeated=n_repeated, 
                           n_classes=2, 
                           n_clusters_per_class=n_clusters_per_class, 
                           weights=None, 
                           flip_y=0.05,
                           class_sep=0.5,
                           hypercube=True,
                           shift=0.0,
                           scale=1.0,
                           shuffle=True,
                           random_state=None)

def load_credit_fraud_detection(n_obs):
    skip = sorted(random.sample(range(1, n_obs), int(n_obs/2)))
    df = pd.read_csv(DATASET_FILE + 'creditdefault_train.csv',
                    header=0,
                    nrows=n_obs,
                    skiprows=skip)
    global variable_names
    variable_names = ['LIMIT_BAL', 'AGE',
               'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
               'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
               'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',]
    print('Columns kept:', variable_names)    
    return np.array(df[variable_names].apply(pd.to_numeric)), np.array(df.iloc[:, -1])

def load_breast_cancer_diagnostic(n_obs):
    df = pd.read_csv(DATASET_FILE + 'breastcancer.csv', header=None, nrows=n_obs)
    return np.array(df.iloc[:, 2:].apply(pd.to_numeric)), np.array(df.iloc[:, 1].apply(lambda x:int(x=='M')))

def load_tennis_matches_winner_detection(n_obs):
    df = pd.read_csv(DATASET_FILE + 'tennis.csv', header=0, nrows=n_obs)
    global variable_names
    y = df['Result']
    for v in ['Player1', 'Player2', 'Result', 'FNL.1', 'FNL.2',
             'ST1.1', 'ST2.1', 'ST3.1', 'ST4.1', 'ST5.1', 'ST1.2', 'ST2.2', 'ST3.2', 'ST4.2', 'ST5.2', 
               'WNR.1', 'WNR.2', ]:
        del df[v]
    variable_names = df.columns
    print(variable_names)
    return np.array(df.apply(pd.to_numeric)), np.array(y.apply(pd.to_numeric))

def load_news_popularity_prediction(n_obs):
    df = pd.read_csv(DATASET_FILE + 'newspopularity.csv', header=0, nrows=n_obs)
    y = np.array(df.iloc[:, -1].apply(pd.to_numeric))
    y = np.array([int(x>=1400) for x in y])
    print(df.iloc[:,2:-1].columns)
    return np.array(df.iloc[:, 2:-1].apply(pd.to_numeric)), y

def main(dataset, n_obs=1000, 
        n_features=3,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        n_clusters_per_class=2):
    DATASETS_ = {'cancer': load_breast_cancer_diagnostic,
            'credit': load_credit_fraud_detection,
            'news':load_news_popularity_prediction,
            'tennis': load_tennis_matches_winner_detection,
            'generate': generate_data}
    if dataset == 'generate':
        
        X, y = DATASETS_[dataset](n_obs, n_features, n_informative, n_redundant, n_repeated, n_clusters_per_class)
    else:
        X, y = DATASETS_[dataset](n_obs)
    X = X - X.mean(axis=0)
    X = X / list(map(max, zip(abs(X.max(axis=0)), abs(X.min(axis=0)))))
    X = np.array(pd.DataFrame(X).fillna(0))
    print(y.mean())
    return X, y

if __name__ == '__main__':
    main('generate', 1000)
