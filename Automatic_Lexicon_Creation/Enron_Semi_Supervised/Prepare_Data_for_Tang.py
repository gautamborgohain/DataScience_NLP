'''
GB
'''

import pandas as pd
import numpy as np
import json
import re
import pickle

variables = json.loads(open('/Volumes/Data/NLP/Utilities/variables.json').read())


def write(df,basedir):
    poscount = 0
    negcount = 0
    neg_sens = df[df.Sentiment == -1].Sentences
    pos_sens = df[df.Sentiment == 1].Sentences[:len(neg_sens)]
    #190357
    print('Total {} sentences to be written to drive'.format(len(pos_sens)))

    for sen in pos_sens:
        f = open(basedir + 'emoticon.pos.' + str(poscount) + '.txt', mode='w')
        f.write(sen)
        f.close()
        poscount += 1

    for sen in neg_sens:
        f = open(basedir + 'emoticon.neg.' + str(negcount) + '.txt', mode='w')
        f.write(sen)
        f.close()
        negcount += 1

    print('Completed writing the sentences to drive.')


if __name__ == '__main__':

    # Write PCF
    pcf_pred_path = variables['pcf_pred_path']
    df = pd.read_csv(pcf_pred_path)

    basedir = '/Volumes/Data/NLP/Automatic_Lexicon_Creation/Enron_Lexicon/Data_official_PCF/'
    write(df,basedir)

    #
    # # Write Subj_Pol two step classified sentences
    # subj_pol_pred_path = variables['subj_pol_pred_path']
    # df = pd.read_csv(subj_pol_pred_path)
    #
    # basedir = '/Volumes/Data/NLP/Automatic_Lexicon_Creation/Enron_Lexicon/Data_official_SUBJ_POL/'
    # write(df,basedir)