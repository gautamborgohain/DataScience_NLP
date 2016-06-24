'''
GB
'''
import json
import pickle
import pandas as pd
from nltk import TreebankWordTokenizer
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC

from Utilities import Text_Features_Pipeline as feats

variables = json.loads(open('/Volumes/Data/NLP/Utilities/variables.json').read())
path = variables['path']

class polarity_classification():


    def train_test(self,X,y):
        y = list(y)
        self.X, self.Xt, self.y, self.yt = train_test_split(X, y, test_size=0.33, random_state=1)

        polarity_pipeline = Pipeline([
            ('features', FeatureUnion([
                ('bow', CountVectorizer(stop_words='english', lowercase=True, ngram_range=(1, 4),
                                        tokenizer=TreebankWordTokenizer().tokenize)),
                ('tfbow', Pipeline([
                    ('tempvect', CountVectorizer()),
                    ('tfvect', TfidfTransformer())
                ])),
                ('cow', CountVectorizer(analyzer='char_wb', lowercase=True, ngram_range=(2, 2))),
                ('lex', feats.HL_lexiconSent_pipeline()),
                ('punc', feats.Punctuations_Pipeline()),
                ('len', feats.Lengths_pipeline()),
                ('elongs', feats.Elongation_pipeline()),
                ('wikiLex', feats.Wiki_Lex_Feats_pipeline()),

            ])
             ),
            ('svm', LinearSVC(C=0.1, max_iter = 3000,class_weight = {0: 1, 1: 1.001, -1: 1.001}))
        ])

        polarity_pipeline.fit(self.X, self.y)
        pickle.dump(polarity_pipeline, open(path + '/polarity_classification_pipeline.pck', mode='wb'))
        print('Printing results on the split Test set : ')
        self.classify(self.Xt, self.yt)

    def classify(self,X,y= None):
        polarity_pipeline = pickle.load(open(path + '/polarity_classification_pipeline.pck', mode='rb'))
        predictions = polarity_pipeline.predict(X)

        if(type(y) != 'NoneType'):
            y = list(y)
            print(accuracy_score(predictions, y))
            print(classification_report(y, predictions))
            print(confusion_matrix(y, predictions))
            print(f1_score(predictions, y, average='macro'))
            print(f1_score(predictions, y, average='macro', labels=[-1, 1]))

        return predictions

if __name__ == '__main__':

    pcf = polarity_classification()

    # data = pd.read_pickle(path + '/data.pck')
    # pcf.train_test(data.Sentences,data.Sentiment)

    data = pd.read_pickle(path + '/test_data.pck')
    pcf.classify(data.Sentences,data.Sentiment)