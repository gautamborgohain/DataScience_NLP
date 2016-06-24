'''
GB
'''
import pickle
from cmath import polar

from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,classification_report
import pandas as pd
import json

class subj_polarity_classification:

    def __init__(self,modelspath, data):

        self.polarity_pipeline = pickle.load(open(modelspath + '/polarity_pipeline.pck', mode = 'rb'))
        self.subjectivity_pipeline = pickle.load(open(modelspath + '/subjectivity_pipeline.pck', mode = 'rb'))
        self.data = data



    def classify(self,test = False):

        subj_predictions = self.subjectivity_pipeline.predict(self.data.Sentences)
        self.data['SUBJ_PRED'] = subj_predictions

        subjective_data = self.data[self.data.SUBJ_PRED == 1]
        polarity_prediction = self.polarity_pipeline.predict(subjective_data.Sentences)
        subjective_data['POLARITY_PRED'] = polarity_prediction

        final = pd.merge(self.data,subjective_data[['ID','POLARITY_PRED']],left_on = 'ID',right_on = 'ID', how = 'left')
        final.POLARITY_PRED.fillna(0, inplace = True)

        preds = [int(val) for val in final.POLARITY_PRED]

        if(test):
            true = [int(val) for val in final.Sentiment]

            print(accuracy_score(preds, true))
            print(classification_report(preds,true))
            print(f1_score(preds,true,average='macro'))
            print(confusion_matrix(preds,true))
            print(f1_score(preds, true, average='macro', labels=[-1, 1]))

        return preds


if __name__ == '__main__':

    variables = json.loads(open('/Volumes/Data/NLP/Utilities/variables.json').read())
    path = variables['path']
    data = pd.read_pickle(path + '/test_data.pck')
    subj_class = subj_polarity_classification(path,data)
    subj_class.classify(True)
