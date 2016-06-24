'''
GB
'''
import json
import re
import numpy as np
import pandas as pd
from email.parser import Parser
from nltk import sent_tokenize
from Automatic_Lexicon_Creation.Enron_Semi_Supervised.polarilty_classification import polarity_classification
from Automatic_Lexicon_Creation.Enron_Semi_Supervised.subj_polarity_classification import subj_polarity_classification

variables = json.loads(open('/Volumes/Data/NLP/Utilities/variables.json').read())
enron_main_path = variables['enron_main_path']


if __name__ == '__main__':

    '''
    Code to read the enron data, extract sentences and save them in a csv file

    '''
    enron = pd.read_pickle(enron_main_path)


    enron.head()
    len(enron)

    p_enron = enron.loc[np.random.permutation(enron.index)]
    p_enron.head()
    len(p_enron)

    del enron

    def cleanSent(sent):
        sent = re.sub(r'\s+', ' ', sent)
        sent = re.sub(r'\?+', '?', sent)
        return sent

    p = Parser()

    sentences = []
    mids = []
    for ind,row in p_enron.iterrows():
        body = row.body
        mid = row.Message_ID
        msg = p.parsestr(body)
        msg = msg._payload
        try:
            e_20 = re.findall(r'=20|=09', msg)

            if len(e_20) > 2:
                continue
            if re.search(r'-+ ?Forwarded by|-+ ?Original Message',msg):
                continue

            lines = sent_tokenize(msg)
            for sent in lines:
                sent = cleanSent(sent)
                if len(sent) > 15 and not (re.search(r'__+|-+', sent)):
                    sentences.append(sent)
                    mids.append(mid)
        except Exception as e:
            print(e)

    df = pd.DataFrame({'MID':mids,'Sentences' : sentences})
    df['ID'] = range(0,len(df))
    df.to_csv('/Users/gautamborgohain/Desktop/enron_sentences.csv')

    print(len(df))
    del p_enron

    '''
        Code to classify the sentences
    '''

    df = pd.read_csv('/Users/gautamborgohain/Desktop/enron_sentences.csv')

    # The single polarity classifer
    sentences = list(df.Sentences)
    pcf = polarity_classification()
    pcf_preds = pcf.classify(df.Sentences)
    pcf_preds_df = pd.DataFrame({'Sentences':sentences,'Sentiment':pcf_preds})
    pcf_preds_path= variables['pcf_preds_path']
    pcf_preds_df.to_csv(pcf_preds_path)


    # The two step classifier
    path = variables['path']
    subj_class = subj_polarity_classification(path, df)
    subj_pol_preds = subj_class.classify()
    subj_pol_preds_df = pd.DataFrame({'Sentences':sentences,'Sentiment':subj_pol_preds})
    subj_pol_enron_path = variables['subj_pol_enron_path']
    subj_pol_preds_df.to_csv(subj_pol_enron_path)


