import pandas as pd
import numpy as np
import json


# Read data and process the data
data1_loc = '/Users/gautamborgohain/Desktop/RA/enron_sentences.csv'
data2_loc = '/Users/gautamborgohain/Desktop/RA/new_enron_sentences_done.xls'

data1 = pd.read_csv(data1_loc)
data2 = pd.read_excel(data2_loc)
data2 = data2[data2.Sentiment != 'repeated with 305']
data2.dropna(inplace=True, how='all')
data2.drop('Comments', axis=1, inplace=True)
data2.dropna(inplace=True, axis=0)
data2.columns = ['Sentences', 'Sentiment']
data = pd.concat([data1, data2])
data.Sentiment.astype('int')
data['ID'] = range(0, len(data))
print('Total size of the data : ', len(data))
print('Distribution of the data : ', data.Sentiment.value_counts())

del data1, data2


variables = json.loads(open('/Volumes/Data/NLP/Utilities/variables.json').read())
path = variables['path']

data.to_pickle(path + '/data_full.pck')
test_data = data.iloc[np.random.permutation(data.ID)[0:2000]]
test_data.to_pickle(path + '/test_data.pck')
data = data[~data.ID.isin(test_data.ID)]
data.to_pickle(path + '/data.pck')