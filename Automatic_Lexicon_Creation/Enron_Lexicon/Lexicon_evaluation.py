'''
GB
'''
import pandas as pd
import numpy as np
import json
import re
import pickle
from nltk import TreebankWordTokenizer
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from Utilities import Text_Features_Pipeline as feats

variables = json.loads(open('/Volumes/Data/NLP/Utilities/variables.json').read())


class Lexicon_evaluation():


    def train_test(self,X,y,Xt = None,yt = None):
        y = list(y)

        if(type(Xt) == 'NoneType'):
            self.X, self.Xt, self.y, self.yt = train_test_split(X, y, test_size=0.33, random_state=1)
        else:
            self.X = X
            self.Xt = Xt
            self.y = y
            self.yt = yt

        print('BASELINE SCORE')
        basline_pipeline = Pipeline([
            ('features', FeatureUnion([
                ('bow', CountVectorizer(stop_words='english', lowercase=True, ngram_range=(1, 4),
                                        tokenizer=TreebankWordTokenizer().tokenize))
            ])
             ),
            ('svm', LinearSVC())
        ])

        basline_pipeline.fit(self.X, self.y)
        print('-'*80)

        print('Printing results on the split Test set : ')
        self.classify(basline_pipeline,self.Xt, self.yt)

        print('BASE ENRON SCORE')
        base_enron_pipeline = Pipeline([
            ('features', FeatureUnion([
                ('bow', CountVectorizer(stop_words='english', lowercase=True, ngram_range=(1, 4),
                                        tokenizer=TreebankWordTokenizer().tokenize)),
                ('enronLex', feats.Enron_Lex_Feats_pipeline())
            ])
             ),
            ('svm', LinearSVC())
        ])

        base_enron_pipeline.fit(self.X, self.y)
        print('-'*80)
        print('Printing results on the split Test set : ')
        self.classify(base_enron_pipeline,self.Xt, self.yt)



        print('BASE WIKI SCORE')
        base_wiki_pipeline = Pipeline([
            ('features', FeatureUnion([
                ('bow', CountVectorizer(stop_words='english', lowercase=True, ngram_range=(1, 4),
                                        tokenizer=TreebankWordTokenizer().tokenize)),
                ('enronLex', feats.Wiki_Lex_Feats_pipeline())
            ])
             ),
            ('svm', LinearSVC())
        ])

        base_wiki_pipeline.fit(self.X, self.y)
        print('-'*80)
        print('Printing results on the split Test set : ')
        self.classify(base_wiki_pipeline,self.Xt, self.yt)


        print('BASE HL SCORE')
        base_hl_pipeline = Pipeline([
            ('features', FeatureUnion([
                ('bow', CountVectorizer(stop_words='english', lowercase=True, ngram_range=(1, 4),
                                        tokenizer=TreebankWordTokenizer().tokenize)),
                ('enronLex', feats.HL_lexiconSent_pipeline())
            ])
             ),
            ('svm', LinearSVC())
        ])

        base_hl_pipeline.fit(self.X, self.y)
        print('-'*80)
        print('Printing results on the split Test set : ')
        self.classify(base_hl_pipeline,self.Xt, self.yt)


        print('ENRON SCORE')
        enron_pipeline = Pipeline([
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
                ('enronLex', feats.Enron_Lex_Feats_pipeline()),

            ])
             ),
            ('svm', LinearSVC(C=0.1, max_iter = 3000,class_weight = {0: 1, 1: 1.001, -1: 1.001}))
        ])

        enron_pipeline.fit(self.X, self.y)
        # pickle.dump(polarity_pipeline, open(path + '/polarity_classification_pipeline.pck', mode='wb'))
        print('-'*80)
        print('Printing results on the split Test set : ')
        self.classify(enron_pipeline,self.Xt, self.yt)


    def classify(self,pipeline,X,y= None):

        predictions = pipeline.predict(X)

        if(type(y) != 'NoneType'):
            y = list(y)
            print(accuracy_score(predictions, y))
            print(classification_report(y, predictions))
            print(confusion_matrix(y, predictions))
            print(f1_score(predictions, y, average='macro'))
            print('macro - FI_Score (-1 and 1) : ',f1_score(predictions, y, average='macro', labels=[-1, 1]))

        return predictions

if __name__ == '__main__':

    le = Lexicon_evaluation()
    path = variables['path']

    print('Evaluating on held out Test data : ')
    data = pd.read_pickle(path + '/test_data.pck')
    print('Size = ', len(data))
    le.train_test(data.Sentences,data.Sentiment)
    del data

    print('Evaluating on Entire Enron data : ')
    data = pd.read_pickle(path + '/data_full.pck')
    print('Size = ', len(data))
    le.train_test(data.Sentences, data.Sentiment)
    del data

    print('Evaluating on held out Entire SemEval data : ')
    semdf_train = pd.read_pickle(variables['semEval_train_path'])
    semdf_test = pd.read_pickle(variables['semEval_test_path'])
    semdf_test = semdf_test[semdf_test['USER ID'].str.startswith('T13')]
    print('Size = ', len(semdf_train), len(semdf_test))
    le.train_test(semdf_train.Tweet, semdf_train.Sentiment,semdf_test.Tweet, semdf_test.Sentiment)

    # data = pd.read_pickle(path + '/test_data.pck')
    # le.classify(data.Sentences,data.Sentiment)




