'''
GB
'''

import json
import pickle
from sklearn.cross_validation import train_test_split
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import Adagrad
import numpy as np
import re
import pandas as pd

variables = json.loads(open('/Volumes/Data/NLP/Utilities/variables.json').read())


def write_lex_dicts():
    neg_enron_path = variables['neg_enron_path']
    pos_enron_path = variables['pos_enron_path']

    lex_df = pd.DataFrame(columns=['Word', 'Polarity', 'Score'])

    Word = []
    Polarity = []
    Score = []

    f = open(pos_enron_path)
    for line in f.readlines():
        word_prob = re.sub('\n', '', line).split(' ')
        Word.append(word_prob[0])
        Polarity.append(1)
        Score.append(word_prob[1])
    f.close()

    f = open(neg_enron_path)
    for line in f.readlines():
        word_prob = re.sub('\n', '', line).split(' ')
        Word.append(word_prob[0])
        Polarity.append(-1)
        Score.append(word_prob[1])
    f.close()

    lex_df['Word'] = Word
    lex_df['Polarity'] = Polarity
    lex_df['Score'] = [float(score) for score in Score]

    lex_word_score = dict()
    lex_word_polarity = dict()
    for index, row in lex_df.iterrows():
        lex_word_score[row.Word] = row.Score
        lex_word_polarity[row.Word] = row.Polarity

    pickle.dump(lex_word_score,open('/Volumes/Data/NLP/Utilities/Lexicons/Enron_Lex_Scr.pck','wb'))
    pickle.dump(lex_word_polarity,open('/Volumes/Data/NLP/Utilities/Lexicons/Enron_Lex_Pol.pck','wb'))
    print('Completed writing the lexicon dicts fot use.')

def build_lexicon(result_dict, threshold = 0.5):

        final_pos_lexicon = dict()
        final_neg_lexicon = dict()
        for word, probs in result_dict.items():
            for index, prob in zip(range(3), probs):
                if prob >= threshold and re.search(r'^[A-Za-z _-]+$', word):
                    if index == 0:
                        final_pos_lexicon[word] = prob
                    elif index == 1:
                        final_neg_lexicon[word] = prob

        print('Lexicon Size : (pos:neg)',len(final_pos_lexicon), len(final_neg_lexicon))

        final_pos_lexicon = sorted(final_pos_lexicon.items(), key=lambda item: (item[1], item[0]), reverse=True)
        final_neg_lexicon = sorted(final_neg_lexicon.items(), key=lambda item: (item[1], item[0]), reverse=True)
        pos_enron_path = variables['pos_enron_path']
        f = open(pos_enron_path, mode='w')
        for word, prob in final_pos_lexicon:
            f.writelines(word + ' ' + str(prob) + '\n')
        f.close()
        neg_enron_path = variables['neg_enron_path']
        f = open(neg_enron_path, mode='w')
        for word, prob in final_neg_lexicon:
            f.writelines(word + ' ' + str(prob) + '\n')
        f.close()




if __name__ == '__main__':

    seed_polarity_dict = pickle.load(open('/Volumes/Data/NLP/Automatic_Lexicon_Creation/Enron_Lexicon/Lexicons/seed_polarity_dict.pck','rb'))
    w2v_embedding_path = variables['w2v_embedding_path']
    data_dict = pickle.load(open(w2v_embedding_path, 'rb'))
    X = [data_dict[word] for word in seed_polarity_dict]
    Y = [[1 if y == 1 else 0, 1 if y == -1 else 0, 1 if y == 0 else 0] for word, y in seed_polarity_dict.items()]

    all_X = [embd for word, embd in data_dict.items()]#Same as the data_dict
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=1)

    model = Sequential()
    model.add(Dense(64,input_dim=100))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))

    model.add(Dense(64, init='uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))

    model.add(Dense(3))
    model.add(Activation('softmax'))
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    adg = Adagrad(lr=0.1)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adg,
                  metrics=['accuracy'])

    model.fit(X_train, y_train,
              nb_epoch=40,
              batch_size=32)
    score = model.evaluate(X_test, y_test, batch_size=32)
    print(score)

    #predict the rest of the vocabulary
    all_X = np.array(all_X)
    predictions = model.predict(all_X)
    result_dict = dict()
    data_dict_words = [word for word in data_dict]

    for word, probs in zip(data_dict_words, predictions):
        result_dict[word] = probs

    build_lexicon(result_dict,0.8)
    write_lex_dicts()

    # from keras.utils.visualize_util import plot
    #
    # plot(model, to_file='model.png')
