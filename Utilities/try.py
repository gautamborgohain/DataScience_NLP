import re
import pickle
import json
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
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

path = variables['path']
data = pd.read_pickle(path + '/test_data.pck')
print('BASELINE SCORE')
basline_pipeline = Pipeline([
    ('features', FeatureUnion([
        ('bow', CountVectorizer(stop_words='english', lowercase=True, ngram_range=(1, 4),
                                tokenizer=TreebankWordTokenizer().tokenize)),
        ('neg', feats.negations_pipeline())
    ])
     ),
    ('svm', LinearSVC())
])

basline_pipeline.fit(data.Sentences,list(data.Sentiment))



################


from collections import Counter

df = pd.read_csv('/Users/gautamborgohain/Desktop/enron_sentences.csv')
len(df)  # 2336052
all_words = [word for sen in df.Sentences for word in sen.split()]
len(all_words)# 38780125
cleaned_vocab = [word for word in all_words if re.search(r'^[A-Za-z _-]*$', word)]  # To get only the words

len(cleaned_vocab)  # 31695701
del df

def build_dataset(words):
    count = [['<unk>', -1]]
    count.extend(Counter(words).most_common(400000-1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count = unk_count + 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary
data, count, dictionary, reverse_dictionary = build_dataset(cleaned_vocab)

f = open('/Volumes/Data/NLP/Automatic_Lexicon_Creation/Enron_Lexicon/Lexicons/vocab.txt',mode = 'w')
for w,v in dictionary.items():
    f.writelines(w+ ' ' + str(v) + '\n')
f.close()