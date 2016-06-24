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
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from Utilities import Text_CNN_Utils as cnn_utils
from Utilities import w2v_trainer as w2v

variables = json.loads(open('/Volumes/Data/NLP/Utilities/variables.json').read())
path = variables['path']


class polarity_classificationCNN():
    def classify(self, x, y, vocabulary, vocabulary_inv, min_word_count, context, num_epochs, sequence_length,
                 embedding_dim, filter_sizes, num_filters, dropout_prob, hidden_dims, batch_size, val_split
                 , model_variation='CNN-static'):

        if model_variation == 'CNN-non-static' or model_variation == 'CNN-static':
            embedding_weights = w2v.w2v_trainer(x, vocabulary_inv, embedding_dim, min_word_count, context)
            if model_variation == 'CNN-static':
                x = embedding_weights[0][x]
        elif model_variation == 'CNN-rand':
            embedding_weights = None
        # Shuffle data
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_shuffled = x[shuffle_indices]
        y_shuffled = y[shuffle_indices]

        print("Vocabulary Size: {:d}".format(len(vocabulary)))

        # Building model
        # ==================================================
        #
        # graph subnet with one input and one output,
        # convolutional layers concateneted in parallel
        graph = Graph()
        graph.add_input(name='input', input_shape=(sequence_length, embedding_dim))
        for fsz in filter_sizes:
            conv = Convolution1D(nb_filter=num_filters,
                                 filter_length=fsz,
                                 border_mode='valid',
                                 activation='relu',
                                 subsample_length=1)
            pool = MaxPooling1D(pool_length=2)
            graph.add_node(conv, name='conv-%s' % fsz, input='input')
            graph.add_node(pool, name='maxpool-%s' % fsz, input='conv-%s' % fsz)
            graph.add_node(Flatten(), name='flatten-%s' % fsz, input='maxpool-%s' % fsz)

        if len(filter_sizes) > 1:
            graph.add_output(name='output',
                             inputs=['flatten-%s' % fsz for fsz in filter_sizes],
                             merge_mode='concat')
        else:
            graph.add_output(name='output', input='flatten-%s' % filter_sizes[0])

        # main sequential model
        model = Sequential()
        if not model_variation == 'CNN-static':
            model.add(Embedding(len(vocabulary), embedding_dim, input_length=sequence_length,
                                weights=embedding_weights))
        model.add(Dropout(dropout_prob[0], input_shape=(sequence_length, embedding_dim)))
        model.add(graph)
        model.add(Dense(hidden_dims))
        model.add(Dropout(dropout_prob[1]))
        model.add(Activation('relu'))
        model.add(Dense(3))
        model.add(Activation('sigmoid'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])

        # Training model
        # ==================================================
        model.fit(x_shuffled, y_shuffled, batch_size=batch_size,
                  nb_epoch=num_epochs,validation_split=val_split)

        cnn_utils.save_Keras_model(model,'/Volumes/Data/NLP/Automatic_Lexicon_Creation/Enron_Semi_Supervised/Models/CNN_model')

        # model = cnn_utils.load_Keras_model('/Volumes/Data/NLP/Automatic_Lexicon_Creation/Enron_Semi_Supervised/Models/CNN_model')
        # model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        # score = model.evaluate(X, Y, verbose=0)

        # pickle.dump(model,open('/Volumes/Data/NLP/Automatic_Lexicon_Creation/Enron_Semi_Supervised/Models/CNN_model.pck','wb'))
        #
        # score = model.evaluate(Xt, yt, batch_size=32)
        #
        # print('Test Score: ', score[0])
        # print('Test accuracy:', score[1])
        # predictions = model.predict(Xt)
        # flattened_predictions = []
        #
        # for scores in predictions:
        #     if np.argmax(scores) == 0:
        #         flattened_predictions.append(0)
        #     elif np.argmax(scores) == 1:
        #         flattened_predictions.append(-1)
        #     elif np.argmax(scores) == 2:
        #         flattened_predictions.append(1)
        #
        # # y = list(y)
        # print(accuracy_score(flattened_predictions, y_copy))
        # print(classification_report(y_copy, flattened_predictions))
        # print(confusion_matrix(y_copy, flattened_predictions))
        # print(f1_score(flattened_predictions, y_copy, average='macro'))
        # print(f1_score(flattened_predictions, y_copy, average='macro', labels=[-1, 1]))


if __name__ == '__main__':

    data = pd.read_pickle(path + '/data_full.pck')

    x, y, vocabulary, vocabulary_inv = cnn_utils.load_data(data)

    pcf = polarity_classificationCNN()

    # Hyper parameters
    sequence_length = len(x[0])
    embedding_dim = 50
    filter_sizes = (3, 4)
    num_filters = 150
    dropout_prob = (0.25, 0.5)
    hidden_dims = 150

    # Training parameters
    batch_size = 32
    num_epochs = 2
    val_split = 0.33

    # Word2Vec parameters, see train_word2vec
    min_word_count = 1  # Minimum word count
    context = 10  # Context window size

    pcf.classify(x, y, vocabulary, vocabulary_inv, min_word_count, context, num_epochs, sequence_length,
                 embedding_dim, filter_sizes, num_filters, dropout_prob, hidden_dims, batch_size, val_split
                 , model_variation='CNN-static')

    # data = pd.read_pickle(path + '/data.pck')
    # pcf.train_test(data.Sentences,data.Sentiment)
    #
    # data = pd.read_pickle(path + '/data_full.pck')
    # pcf.classify(data.Sentences, data.Sentiment)
