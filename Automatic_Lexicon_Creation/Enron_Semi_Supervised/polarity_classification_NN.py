'''
GB
'''
import json
import pickle
import pandas as pd
import numpy as np
from nltk import TreebankWordTokenizer
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import Adagrad,SGD
from keras.utils import np_utils
from Utilities import Text_Features_Pipeline as feats

variables = json.loads(open('/Volumes/Data/NLP/Utilities/variables.json').read())
path = variables['path']


class polarity_classificationNN():

    def features_pipeline(self,X, y):
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
            # ('svm', LinearSVC())
        ])

        polarity_pipeline.fit(X, y)
        pickle.dump(polarity_pipeline, open(path + '/polarity_classification_pipeline_Features.pck', mode='wb'))
        # print('Printing results on the split Test set : ')
        # self.classify(self.Xt, self.yt)

    def classify(self, X, y):

        y = list(y)
        X, Xt, y, yt = train_test_split(X, y, test_size=0.33, random_state=1)
        y_copy = yt
        self.features_pipeline( X, y)
        polarity_pipeline = pickle.load(open(path + '/polarity_classification_pipeline_Features.pck', mode='rb'))

        X = polarity_pipeline.transform(X)
        Xt = polarity_pipeline.transform(Xt)

        X = X.todense()
        Xt = Xt.todense()

        X = np.array(X)
        Xt = np.array(Xt)

        y = np_utils.to_categorical(y, 3)
        yt = np_utils.to_categorical(yt, 3)

        feature_size = len(X[0])
        model = Sequential()
        model.add(Dense(20, input_dim=feature_size))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        # model.add(Dense(64))
        # model.add(Activation('tanh'))
        # model.add(Dropout(0.5))

        model.add(Dense(3))

        # adg = Adagrad(lr=0.01)
        # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

        model.compile(loss='categorical_crossentropy',
                      # optimizer=adg,
                      optimizer='adam',
                      metrics=['accuracy'])

        model.fit(X, y,
                  nb_epoch=10,
                  batch_size=32)
        score = model.evaluate(Xt, yt, batch_size=32)

        print('Test Score: ', score[0])
        print('Test accuracy:', score[1])
        predictions = model.predict(Xt)
        flattened_predictions = []

        for scores in predictions:
            if np.argmax(scores) == 0:
                flattened_predictions.append(0)
            elif np.argmax(scores) == 1:
                flattened_predictions.append(-1)
            elif np.argmax(scores) == 2:
                flattened_predictions.append(1)

        # y = list(y)
        print(accuracy_score(flattened_predictions, y_copy))
        print(classification_report(y_copy, flattened_predictions))
        print(confusion_matrix(y_copy, flattened_predictions))
        print(f1_score(flattened_predictions, y_copy, average='macro'))
        print(f1_score(flattened_predictions, y_copy, average='macro', labels=[-1, 1]))

        return predictions


if __name__ == '__main__':
    pcf = polarity_classificationNN()

    # data = pd.read_pickle(path + '/data.pck')
    # pcf.train_test(data.Sentences,data.Sentiment)

    data = pd.read_pickle(path + '/data_full.pck')
    pcf.classify(data.Sentences, data.Sentiment)
