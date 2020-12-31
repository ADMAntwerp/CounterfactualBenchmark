import lore

from prepare_dataset import *
from neighbor_generator import *

from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

import re
def generateCf(record2explain, dataset, deltas, blackbox, deltachanges):
    columns = dataset['columns'][1:]
    cfs = []
    for delta in deltas:
        cf = record2explain.copy()
        for k, v in delta.items():
            index = columns.index(k)
            if k in dataset['continuous']:
                if len(re.findall('[>/</=]+', v)) == 1:
                    value = float((re.sub('[>/</=]+', '', v)))
                    if '>' in list(v):
                        cf[index] = value + 0.000001
                    else:
                        cf[index] = value - 0.000001
                else:
                    min_c = float(v.split('<')[0])
                    max_c = float(v.split('<')[2].split('=')[-1])
                    cf[index] = min_c + 0.000001
        deltacats = []
        for delta in deltachanges:
            if delta != {}:
                if list(delta.keys())[0] in dataset['discrete']:
                    deltacats.append(delta)
        if len(deltacats) > 0:
            for delta_cat in deltacats:
                cf_updated = cf.copy()
                for k_cat, v_cat in delta_cat.items():
                    if k_cat not in dataset['continuous']:
                        index = columns.index(k_cat)
                        print(k_cat, v_cat)
                        if v_cat in list(dataset['label_encoder'][k_cat].classes_):
                            value = list(dataset['label_encoder'][k_cat].classes_).index(v_cat)
                            print('#######################', value)
                        elif int(v_cat) in dataset['label_encoder'][k_cat].classes_:
                            value = list(dataset['label_encoder'][k_cat].classes_).index(int(v_cat))
                        elif int(v_cat) in list(int(z) if re.sub('[-]', '', re.sub('[.]', '', str(z), count=1), count=1).isnumeric() else str(z) for z in dataset['label_encoder'][k_cat].classes_):
                            temp_array = list(int(z) if re.sub('[-]', '', re.sub('[.]', '', str(z), count=1), count=1).isnumeric() else str(z) for z in dataset['label_encoder'][k_cat].classes_)
                            value = list(dataset['label_encoder'][k_cat].classes_).index(temp_array.index(int(v_cat)))
                        cf_updated[index] = value
                cfs.append(cf_updated)
                print('################################')
                print('Original:')
                print(blackbox.predict_proba(np.array([record2explain])))
                print('CF:')
                print(blackbox.predict_proba(np.array([cf_updated])))
                print('################################')
        else:
            cfs.append(cf)
            print('################################')
            print('Original:')
            print(blackbox.predict_proba(np.array([record2explain])))
            print('CF:')
            print(blackbox.predict_proba(np.array([cf])))
            print('################################')

    return cfs


def main(df, categorical_columns, class_name, ohc, model, cfData):

    dataset = prepare_generic_dataset(df.append(cfData, ignore_index=True), categorical_columns, class_name)

    record2explain = dataset['X'][dataset['X'].shape[0]-1]

    dataset['X'] = dataset['X'][:dataset['X'].shape[0]-1]
    dataset['y'] = dataset['y'][:dataset['y'].shape[0]-1]

    print(dataset['label_encoder'][dataset['class_name']].classes_)
    print(dataset['possible_outcomes'])

    X, y = dataset['X'], dataset['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    converter = Converter(dataset['df'], ohc, dataset['label_encoder'])

    blackbox = ModelBlackBoxTF(converter, model)

    X2E = X_test
    y2E = blackbox.predict(X2E)
    y2E = np.asarray([dataset['possible_outcomes'][i] for i in y2E])



    explanation, infos = lore.explain(record2explain, X2E, dataset, blackbox,
                                      ng_function=genetic_neighborhood,
                                      discrete_use_probabilities=True,
                                      continuous_function_estimation=False,
                                      returns_infos=True,
                                       sep=';', log=False)

    if len(explanation) > 0:
        dfX2E = build_df2explain(blackbox, X2E, dataset).to_dict('records')
        # dfx = dfX2E[idx_record2explain]
        dfx = record2explain

        # x = build_df2explain(blackbox, X2E[idx_record2explain].reshape(1, -1), dataset).to_dict('records')[0]

        print('x = %s' % dfx)
        print('r = %s --> %s' % (explanation[0][1], explanation[0][0]))
        for delta in explanation[1]:
            print('delta', delta)

        covered = lore.get_covered(explanation[0][1], dfX2E, dataset)
        print(len(covered))
        print(covered)

        print(explanation[0][0][dataset['class_name']], '<<<<')

        def eval(x, y):
            return 1 if x == y else 0

        precision = [1-eval(v, explanation[0][0][dataset['class_name']]) for v in y2E[covered]]
        cfs = generateCf(record2explain, dataset, [explanation[0][1]], blackbox, explanation[1])
        print(precision)
        print(np.mean(precision), np.std(precision))

        return pd.DataFrame(converter.convert(cfs))
    return []

class Converter:

    def __init__(self, df, oh_c, label_encoder):
        # Remove the first column as it is the output column
        self.columns = list(df.columns)[1:]

        self.idx_to_ohidx = {}
        for idx, ohc in enumerate(oh_c):
            if ohc!='output':
                col_name = str(int(ohc.split('_')[0]))
                if (len(ohc.split('_'))>1):
                    col_name+='_'+'_'.join(ohc.split('_')[1:])
                self.idx_to_ohidx[col_name] = idx


        self.out_to_oh = {}
        for c in list(df.columns):
            converter = {}
            if str(c) in label_encoder.keys():
                for idx, fc in enumerate(list(label_encoder[str(c)].classes_)):
                    converter[idx] = fc
                self.out_to_oh[c] = converter

    def convert(self, arr_out):
        np_array_converted = []
        if type(arr_out) != list:
            arr_out = arr_out.tolist()
        for arr in arr_out:
            array_converted = [0]*len(self.idx_to_ohidx.keys())
            for idx, value in enumerate(arr):
                if str(idx) in self.out_to_oh.keys():
                    cats_dict = self.out_to_oh[str(idx)]
                    value_converted = cats_dict[value]
                    if str(idx)+'_'+str(value_converted) in self.idx_to_ohidx.keys():
                        array_converted[self.idx_to_ohidx[str(idx)+'_'+str(value_converted)]] = 1
                    elif str(idx)+'_'+str(int(value_converted)) in self.idx_to_ohidx.keys():
                        array_converted[self.idx_to_ohidx[str(idx) + '_' + str(int(value_converted))]] = 1
                    else:
                        array_converted[self.idx_to_ohidx[str(idx)]] = value_converted
                else:
                    array_converted[self.idx_to_ohidx[str(idx)]] = value
            np_array_converted.append(array_converted)

        return np.array(np_array_converted)

# from test.CFmodel import CreateModel

class ModelBlackBoxTF:
    def __init__(self, converter, model):
        self.converter = converter

        self.model = model

    def fit(self, traning_data, labels):
        c_traning_data = self.converter.convert(traning_data)
        self.model.fit(c_traning_data, labels, epochs=100)

    def predict(self, predict_data):
        c_predict_data = self.converter.convert(predict_data)
        output_labels = self.model.predict(c_predict_data)

        return pd.DataFrame(output_labels)[0].map(int).to_numpy()

    def predict_proba(self, predict_data):
        c_predict_data = self.converter.convert(predict_data)
        output_labels = self.model.predict(c_predict_data)

        return output_labels

if __name__ == "__main__":
    # dataset_name - The name of the dataset
    # path_data - The path to reach the data
    # categorical_columns - List with the name of all categorical features
    # class_name - The name of the output/target class


    print('a')
    ohc = list(''.join(c.split(' ')) for c in pd.read_csv('./test/OH_NORM_Adult.csv').columns)

    df = pd.read_csv('./test/NORM_Adult.csv')

    main(df=df,
         categorical_columns=['0', '1', '2', '3', '4', '5', '7'],
         class_name='output',
         ohc=ohc,
         model=CreateModel('Adult', '0'),
         cfData=df.loc[100]
         )

