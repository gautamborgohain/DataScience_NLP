'''
GB
'''
import re
import pickle
import json
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin

variables = json.loads(open('/Volumes/Data/NLP/Utilities/variables.json').read())

class Elongation_pipeline(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, frame):
        df = pd.DataFrame()
        no_elong = []
        avg_elong = []
        reg = re.compile('([a-zA-Z])\\1{3,}')
        for tweet in frame:
            elongs = []
            no_elong.append(len(reg.findall(tweet)))
            for match in reg.finditer(tweet):
                elongs.append((match.end() - match.start()))

            avg_elong.append(np.mean(elongs))
        df['No_Elong'] = no_elong
        df['Avg_Elong'] = avg_elong
        df.fillna(value=0, inplace=True)
        return df.as_matrix()


class Emoticons_pipeline(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, frame):
        df = pd.DataFrame()
        posEmot, negEmot, posEmo, negEmo, neutralEmo = [], [], [], [], []
        for tweet in frame:
            posEmot.append(len(re.findall(r':-D|:D|:-\)|:\)|;-\)|;\)', tweet)))
            negEmot.append(len(re.findall(r':-\(|:\(', tweet)))

        df['EMOT_POS'] = posEmot
        df['EMOT_NEG'] = negEmot

        return df.as_matrix()


class HL_lexiconSent_pipeline(BaseEstimator, TransformerMixin):

    def __init__(self):
        HL_path = variables['HL_path']
        HL_lex = pickle.load(open(HL_path, 'rb'))
        self.poslist = HL_lex['poslist']
        self.neglist = HL_lex['neglist']

    def getPositiveWordCount(self,tweet):
        lmtzr = WordNetLemmatizer()
        countPos = 0
        for word in word_tokenize(tweet):
            word = word.lower()
            if len(word) >= 2 and word in self.poslist:
                countPos += 1
            elif len(word) >= 2 and lmtzr.lemmatize(word) in self.poslist:
                countPos += 1
        return countPos

    def getNegativeWordCount(self,tweet):
        lmtzr = WordNetLemmatizer()
        countNeg = 0
        for word in word_tokenize(tweet):
            word = word.lower()
            if len(word) >= 2 and word in self.neglist:
                countNeg += 1
            elif len(word) >= 2 and lmtzr.lemmatize(word) in self.poslist:
                countNeg += 1

        return countNeg

    def fit(self, x, y=None):
        return self

    def transform(self, frame):
        df = pd.DataFrame(columns=['POS_LEX', 'NEG_LEX'])
        df['POS_LEX'] = [self.getPositiveWordCount(tweet) for tweet in frame]
        df['NEG_LEX'] = [self.getNegativeWordCount(tweet) for tweet in frame]

        return df.as_matrix()


class Lengths_pipeline(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, frame):
        df = pd.DataFrame()
        df['Length'] = [len(tweet) for tweet in frame]
        df['Length_words_MAX'] = [max([len(part) for part in tweet.split(' ')]) for tweet in frame]
        df['Length_words_MIN'] = [max([len(part) for part in re.sub(r'\s+', ' ', tweet).split(' ')]) for tweet in frame]
        df['Length_words_AVG'] = [np.mean([len(part) for part in tweet.split(' ')]) for tweet in frame]
        return df.as_matrix()


class negations_pipeline(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, frame):
        df = pd.DataFrame()
        dons = []
        nots = []
        negations = []
        negates_re = re.compile(
            "aint | arent | cannot | cant | couldnt | darent | didnt | doesnt | ain't | aren't | can't | couldn't | daren't | didn't | doesn't | dont | hadnt | hasnt | havent | isnt | mightnt | mustnt | neither | don't | hadn't | hasn't | haven't | isn't | mightn't | mustn't | neednt | needn't | never | none | nope | nor | not | nothing | nowhere | oughtnt | shant | shouldnt | uhuh | wasnt | werent | oughtn't | shan't | shouldn't | uh-uh | wasn't | weren't | without | wont | wouldnt | won't | wouldn't | rarely | seldom | despite")

        for tweet in frame:
            dons.append(len(re.findall(' don | dont | don\'t | never | no | not | neither | nor | none ', tweet)))
            nots.append(len(re.findall(' not ', tweet)))
            negations.append(len(re.findall(negates_re, tweet)))

        df = pd.DataFrame({'dons': dons, 'nots': nots, 'negates' : negations})
        return df.as_matrix()


class otherFeats_pipeline(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, frame):
        df = pd.DataFrame()
        no_of_sents = []
        no_of_newlines = []
        no_of_rsdots = []
        no_of_rspaces = []
        for tweet in frame:
            no_of_sents.append(len(tweet.split('.')))
            no_of_newlines.append(len(re.findall(r'[\n]', tweet)))
            no_of_rsdots.append(len(re.findall('[\.]+', tweet)))
            no_of_rspaces.append(len(re.findall('[\s]+', tweet)))

        df['SENTS'] = no_of_sents
        df['NEWLINES'] = no_of_newlines
        df['DOTS'] = no_of_rsdots
        df['SPACES'] = no_of_rspaces
        return df.as_matrix()


class Punctuations_Pipeline(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, frame):
        df = pd.DataFrame()
        punc_excl = []
        punc_ques = []
        punc_quot = []
        for tweet in frame:
            punc_excl.append(len(re.findall(r'!', tweet)))
            punc_ques.append(len(re.findall(r'\?', tweet)))
            punc_quot.append(len(re.findall(r'\'|"', tweet)))

        df['PUNC_EXCL'] = punc_excl
        df['PUNC_QUES'] = punc_ques
        df['PUNC_QUOT'] = punc_quot
        return df.as_matrix()


class Wiki_Lex_Feats_pipeline(BaseEstimator, TransformerMixin):
    def __init__(self):

        self.lex_word_score = pickle.load(open('/Volumes/Data/NLP/Utilities/Lexicons/Wiki_Lex_Scr.pck','rb'))
        self.lex_word_polarity = pickle.load( open('/Volumes/Data/NLP/Utilities/Lexicons/Wiki_Lex_Pol.pck','rb'))

    def fit(self, x, y=None):
        return self

    def transform(self, frame):
        predictions = []
        for tweet in frame:
            words = word_tokenize(tweet)
            pos_word_count = 0
            neg_word_count = 0
            pos_score = []
            neg_score = []
            for word in words:
                polarity = self.lex_word_polarity.get(word, 0)
                if polarity == 1:
                    pos_word_count += 1
                    pos_score.append(self.lex_word_score.get(word))
                elif polarity == -1:
                    neg_word_count += 1
                    neg_score.append(self.lex_word_score.get(word))
            max_pos_score = max(pos_score) if len(pos_score) > 0 else 0
            max_neg_score = max(neg_score) if len(neg_score) > 0 else 0
            las_word_pos = self.lex_word_score.get(words[-1]) if self.lex_word_polarity.get(words[-1]) == 1 else 0
            las_word_neg = self.lex_word_score.get(words[-1]) if self.lex_word_polarity.get(words[-1]) == -1 else 0
            predictions.append(
                [pos_word_count, neg_word_count, sum(pos_score), sum(neg_score), max_pos_score, max_neg_score,
                 las_word_pos, las_word_neg])

        return np.matrix(predictions)

class Enron_Lex_Feats_pipeline(BaseEstimator, TransformerMixin):
    def __init__(self):

        self.lex_word_score = pickle.load(open('/Volumes/Data/NLP/Utilities/Lexicons/Enron_Lex_Scr.pck', 'rb'))
        self.lex_word_polarity = pickle.load(open('/Volumes/Data/NLP/Utilities/Lexicons/Enron_Lex_Pol.pck', 'rb'))

    def fit(self, x, y=None):
        return self

    def transform(self, frame):
        predictions = []
        for tweet in frame:
            words = word_tokenize(tweet)
            pos_word_count = 0
            neg_word_count = 0
            pos_score = []
            neg_score = []
            for word in words:
                polarity = self.lex_word_polarity.get(word, 0)
                if polarity == 1:
                    pos_word_count += 1
                    pos_score.append(self.lex_word_score.get(word))
                elif polarity == -1:
                    neg_word_count += 1
                    neg_score.append(self.lex_word_score.get(word))
            max_pos_score = max(pos_score) if len(pos_score) > 0 else 0
            max_neg_score = max(neg_score) if len(neg_score) > 0 else 0
            las_word_pos = self.lex_word_score.get(words[-1]) if self.lex_word_polarity.get(words[-1]) == 1 else 0
            las_word_neg = self.lex_word_score.get(words[-1]) if self.lex_word_polarity.get(words[-1]) == -1 else 0
            predictions.append(
                [pos_word_count, neg_word_count, sum(pos_score), sum(neg_score), max_pos_score, max_neg_score,
                 las_word_pos, las_word_neg])

        return np.matrix(predictions)