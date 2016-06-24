'''
GB
'''
import pandas as pd
import numpy as np
import json
import re
import pickle

from theano.gof.utils import comm_guard

variables = json.loads(open('/Volumes/Data/NLP/Utilities/variables.json').read())

if __name__ == '__main__':

    w2v_embedding_path = variables['w2v_embedding_path']
    data_dict = pickle.load(open(w2v_embedding_path, 'rb'))
    vocab = [key for key, val in data_dict.items()]
    #179,417
    HL_path = variables['HL_path']
    HL_lex = pickle.load(open(HL_path, 'rb'))
    poslist = HL_lex['poslist']
    neglist = HL_lex['neglist']
    print('HL Lex : ', len(poslist), len(neglist))

    subjLexLoc = '/Users/gautamborgohain/PycharmProjects/Twitter_target_dependent_SA/subjectivity.csv'
    subjLex = pd.read_csv(subjLexLoc)
    neg_subj_list = []
    pos_subj_list = []
    neu_subj_list = []
    for index, row in subjLex.iterrows():
        if row.priorpolarity == 'negative':
            neg_subj_list.append(row.word1)
        elif row.priorpolarity == 'positive':
            pos_subj_list.append(row.word1)
        elif row.priorpolarity == 'neutral' :
            neu_subj_list.append(row.word1)


    print('MPQA Lex : ', len(pos_subj_list), len(neg_subj_list), len(neu_subj_list))

    vader_lex_path = variables['vader_lex_path']
    vader_lex = pickle.load(open(vader_lex_path,'rb'))
    vader_threshold = 1.0
    vader_pos = [word for word,score in vader_lex.items() if score>vader_threshold]
    vader_neg = [word for word,score in vader_lex.items() if score<-vader_threshold]
    vader_neu = [word for word,score in vader_lex.items() if score>=-vader_threshold and score<=vader_threshold]
    print('Vader Lex : ', len(vader_pos),len(vader_neg), len(vader_neu))
    cleaned_vocab = [word for word in vocab if re.search(r'^[A-Za-z _-]*$', word)]  # To get only the words
    #63,912

    pos_seeds = []
    neg_seeds = []
    neu_seeds = []


    common_vocab = pickle.load(open('/Volumes/Data/NLP/Automatic_Lexicon_Creation/Enron_Lexicon/Lexicons/enron_common.txt','rb'))
    common_vocab = [word for word,count in common_vocab]

    for seed in common_vocab:
        seed_lower = seed.lower()
        if seed_lower in poslist or seed_lower in pos_subj_list or seed_lower in vader_pos:
            pos_seeds.append(seed)
        elif seed_lower in neglist or seed_lower in neg_subj_list or seed_lower in vader_neg:
            neg_seeds.append(seed)
        elif seed_lower in neu_subj_list or seed_lower in vader_neu:
            neu_seeds.append(seed)
        # if len(pos_seeds) > 1000 or len(neg_seeds) > 1000:
        #     break
        # else:
        #     neu_seeds.append(seed)

    print('Seeds found in Lexicons (pos:neg:neu): ', len(pos_seeds), len(neg_seeds), len(neu_seeds))
    #4467 5168 1545 without the stopping condition of 1000 ; when selecting all the words in the vocabulary
    #3124 3157 1124 when selecting only the 100000 most common words in the dataset
    # neu_seeds_sample = np.random.permutation(neu_seeds)[0:3000]

    f = open('/Volumes/Data/NLP/Automatic_Lexicon_Creation/Enron_Lexicon/Lexicons/pos_seeds.txt', mode='w')
    for word in pos_seeds:
        f.write(word + '\n')
    f.close()
    f = open('/Volumes/Data/NLP/Automatic_Lexicon_Creation/Enron_Lexicon/Lexicons/neg_seeds.txt', mode='w')
    for word in neg_seeds:
        f.write(word + '\n')
    f.close()
    f = open('/Volumes/Data/NLP/Automatic_Lexicon_Creation/Enron_Lexicon/Lexicons/neu_seeds.txt', mode='w')
    for word in neu_seeds:
        f.write(word + '\n')
    f.close()
    ##################################################
    pos_seeds = [re.sub('\n','',line) for line in open('/Volumes/Data/NLP/Automatic_Lexicon_Creation/Enron_Lexicon/Lexicons/pos_seeds.txt', mode='r')]
    neg_seeds = [re.sub('\n','',line) for line in open('/Volumes/Data/NLP/Automatic_Lexicon_Creation/Enron_Lexicon/Lexicons/neg_seeds.txt', mode='r')]
    neu_seeds = [re.sub('\n','',line) for line in open('/Volumes/Data/NLP/Automatic_Lexicon_Creation/Enron_Lexicon/Lexicons/neu_seeds.txt', mode='r')]

    polarity_dict = dict()
    for word in pos_seeds:
        polarity_dict[word] = 1
    for word in neg_seeds:
        polarity_dict[word] = -1
    for word in neu_seeds:
        polarity_dict[word] = 0

    pickle.dump(polarity_dict,open('/Volumes/Data/NLP/Automatic_Lexicon_Creation/Enron_Lexicon/Lexicons/seed_polarity_dict.pck','wb'))

    # udDict_path = variables['udDict_path']
    # udDict = pickle.load(open(udDict_path, 'rb'))
    # expnd_pos_words = [expansion for word in pos_seeds if udDict.get(word) for expansion in udDict.get(word) if data_dict.get(expansion) != None]
    # expnd_neg_words = [expansion for word in neg_seeds if udDict.get(word) for expansion in udDict.get(word) if data_dict.get(expansion) != None]
    #
    # print('Expansion Positive words : ', len(expnd_pos_words))
    # print('Expansion Negative words : ', len(expnd_neg_words))
    # pos_seeds.extend(np.random.permutation(expnd_pos_words))
    # neg_seeds.extend(np.random.permutation(expnd_neg_words))
    #
    # print(len(pos_seeds), len(neg_seeds), len(neu_seeds_sample))
    #
    # ########################################################
    #
    # vpath = '/Users/gautamborgohain/Downloads/vaderSentiment-0.5/vaderSentiment/vader_sentiment_lexicon.txt'
    # f = open('/Users/gautamborgohain/Downloads/vaderSentiment-0.5/vaderSentiment/vader_sentiment_lexicon.txt', 'rb')
    #
    #
    # def make_lex_dict(f):
    #     return dict(map(lambda wm: (wm[0], float(wm[1])), [wmsr.decode('ascii','ignore').strip().split('\t')[0:2] for wmsr in open(f,'rb')]))
    #
    # lex_dict = make_lex_dict(vpath)
    #
    # pickle.dump(lex_dict,open('/Volumes/Data/NLP/Utilities/Lexicons/Vader_Lex','wb'))