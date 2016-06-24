import json
import pickle

import pandas as pd
from nltk import TreebankWordTokenizer
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer , TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC

from Utilities import Text_Features_Pipeline as feats

variables = json.loads(open('/Volumes/Data/NLP/Utilities/variables.json').read())
path = variables['path']

# Clasifier for Polarity

data = pd.read_pickle(path + '/data.pck')

data = data[data.Sentiment.isin([-1, 1])]  # only the objective sentences

X, Xt, y, yt = train_test_split(data.Sentences, data.Sentiment, test_size=0.33, random_state=1)
y = list(y)
yt = list(yt)

polarity_pipeline = Pipeline([
    ('features', FeatureUnion([
        ('bow', CountVectorizer(stop_words='english', lowercase=True, ngram_range=(1, 4),
                                tokenizer=TreebankWordTokenizer().tokenize)),
        ('lex', feats.HL_lexiconSent_pipeline()),
        ('punc', feats.Punctuations_Pipeline()),
        ('len', feats.Lengths_pipeline()),
        ('elongs', feats.Elongation_pipeline()),
        ('wikiLex', feats.Wiki_Lex_Feats_pipeline()),

    ])
     ),
    ('svm', LinearSVC(C=0.1))
])

polarity_pipeline.fit(X, y)
predictions = polarity_pipeline.predict(Xt)
print('-' * 30)
print('Building Polarity Model')

print(accuracy_score(predictions, yt))

print(classification_report(yt, predictions))
print(confusion_matrix(yt, predictions))
print(f1_score(predictions, yt, average='macro'))
print(f1_score(predictions, yt, average='macro', labels=[-1, 1]))

# Classifier for Objectivity and Subjectivity
data = pd.read_pickle(path + '/data.pck')

X, Xt, y, yt = train_test_split(data.Sentences, data.Sentiment, test_size=0.33, random_state=1)

y = list(map(lambda x: x if x != -1 else 1, y))
yt = list(map(lambda x: x if x != -1 else 1, yt))

print('-' * 30)
print('Building Subjectivity Model')

subjectivity_pipeline = Pipeline([
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
    ('svm', LogisticRegression(C = 2, solver='liblinear'))
])

subjectivity_pipeline.fit(X, y)
predictions = subjectivity_pipeline.predict(Xt)
print(accuracy_score(predictions, yt))
print(classification_report(yt, predictions))
print(f1_score(predictions, yt, average='macro'))
print(confusion_matrix(yt, predictions))

pickle.dump(polarity_pipeline, open(path + '/polarity_pipeline.pck', mode='wb'))
pickle.dump(subjectivity_pipeline, open(path + '/subjectivity_pipeline.pck', mode='wb'))

#Test
polarity_pipeline = pickle.load(open(path + '/polarity_pipeline.pck', mode='rb'))
subjectivity_pipeline = pickle.load(open(path + '/subjectivity_pipeline.pck', mode='rb'))

