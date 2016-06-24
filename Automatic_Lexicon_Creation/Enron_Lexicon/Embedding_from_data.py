'''
GB
'''
import pandas as pd
import numpy as np
import json
import re
import logging
import matplotlib.pyplot as plt
import pickle
from sklearn.manifold import TSNE
import gensim

variables = json.loads(open('/Volumes/Data/NLP/Utilities/variables.json').read())
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def sen_permutation(sens):
    return np.random.permutation(sens)



def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)


if __name__ == '__main__':

    df = pd.read_csv('/Users/gautamborgohain/Desktop/enron_sentences.csv')
    len(df)  # 2336052
    all_words = [word for sen in df.Sentences for word in sen.split()]
    len(all_words) #38780125
    words  =  set(all_words)
    len(words) # 5107253

    #
    # from collections import Counter
    # c = Counter(all_words)
    # most_common = c.most_common(100000)
    # pickle.dump(most_common,open('/Volumes/Data/NLP/Automatic_Lexicon_Creation/Enron_Lexicon/Lexicons/enron_common.txt','wb'))
    #


    # df = df[0:500000]
    # words  =  set([word for sen in df.Sentences for word in sen.split()])
    # len(words) # 277533


    sens = list(df.Sentences)
    sens = [[s for s in sen.split()] for sen in sens]



    print('Building w2v vocabulary : ')
    model = gensim.models.word2vec.Word2Vec(sens, min_count=3, sg=1, size=100, alpha=0.025, window=3, seed=1)
    print('Completed training w2v model')
    vocab = model.vocab
    print('Vocabulary size : ', len(vocab))
    path = variables['path']
    model.save(path + '/enron_w2v_gensim.model')
    model.save_word2vec_format(path + '/text.model.bin', binary=True)


    model = gensim.models.Word2Vec.load_word2vec_format('/Volumes/Data/NLP/Automatic_Lexicon_Creation/Enron_Lexicon/Embeddings/vectors_enron_50_H9.txt')
    print('Similar man',model.most_similar('man'))
    print('Similar good',model.most_similar('good'))
    print('Similar earth',model.most_similar('earth'))
    print('Similar apple',model.most_similar('apple'))


    # # W2V embedding
    # model = gensim.models.Word2Vec(max_vocab_size= 50000,min_count=1, sg=1, size=50, alpha=0.025, window=2, seed=1, workers=5)
    # print('Building w2v vocabulary : ')
    # model.build_vocab(sens)
    # vocab = model.vocab
    # print('Vocabulary size : ', len(vocab))
    # epoch = 10
    # del df
    # print('Straining w2v model training: ')
    # for ep in range(epoch):o
    #     print('Epoch : ',ep)
    #     model.train(sen_permutation(sens))
    #
    # print('Completed training w2v model')
    #
    # path = variables['path']
    # model.save(path + '/enron_w2v_gensim.model')
    # model.save_word2vec_format(path + '/text.model.bin', binary=True)

    model = gensim.models.Word2Vec.load(path + '/enron_w2v_gensim.model')

    w2v_embedding = dict()
    for word in vocab:
        w2v_embedding[word] = model[word]
    w2v_embedding_path = variables['w2v_embedding_path']
    pickle.dump(w2v_embedding,open(w2v_embedding_path,'wb'))

    print('Building TSNE manifold graph of the embeddings :')
    try:
        tsne = TSNE(learning_rate=300, init='pca', n_iter=5000)
        plot_only = 500
        vocab=[word for word in vocab]
        labels = vocab[:plot_only]
        low_dim_embs = tsne.fit_transform(model[labels])
        plot_with_labels(low_dim_embs, labels,'tsne_w2v.png')

    except Exception as e:
        print(e)
