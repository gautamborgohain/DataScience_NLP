'''
GB
'''
import json
import re
import numpy as np
import tensorflow as tf
from sklearn.cross_validation import train_test_split
import pickle

variables = json.loads(open('/Volumes/Data/NLP/Utilities/variables.json').read())

class Softmax_Lexicon():

    def accuracy(self,labels, predictions):
        return 100 * (np.sum(np.argmax(labels, 1) == np.argmax(predictions, 1)) / predictions.shape[0])


    def train(self,X_train, X_test, y_train, y_test,all_X,data_dict,size = 50, epochs = 1001):

        print('Preparing Tensorflow graph...')
        graph = tf.Graph()
        with graph.as_default():
            x = tf.placeholder(tf.float32, [None, size])
            x_test = tf.placeholder(tf.float32, [None, size])
            # all_words = tf.constant(all_X)
            W = tf.Variable(tf.zeros([size, 3]))
            # W = tf.Variable(tf.random_uniform([size, 3]))
            b = tf.Variable(tf.zeros([3]))
            logits = tf.matmul(x, W) + b
            y_ = tf.placeholder(tf.float32, [None, 3])
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y_))
            optimizer = tf.train.AdagradOptimizer(0.01).minimize(loss)
            training_predictions = tf.nn.softmax(logits)
            testing_predictions = tf.nn.softmax(tf.matmul(x_test, W) + b)
            # all_predictions = tf.nn.softmax(tf.matmul(all_words, W) + b)

        print('Starting to Train..')
        with tf.Session(graph=graph) as session:
            tf.initialize_all_variables().run()
            for step in range(epochs):
                feed_dict = {x: X_train, y_ : y_train, x_test: X_test}
                l, _, tr_predictions, test_predictions = session.run(
                    [loss, optimizer, training_predictions, testing_predictions], feed_dict=feed_dict)
                if step % 100 == 0:
                    print('Loss at Step ', step, ' : ', l)
                    print('Training Accuracy : ', self.accuracy(y_train, tr_predictions))
                    print('Testing Accuracy : ', self.accuracy(y_test, test_predictions))

            # all_preds = session.run([all_predictions])

        result_dict = dict()
        data_dict_words = [word for word in data_dict]

        for word, probs in zip(data_dict_words, all_preds):
            result_dict[word] = probs

        return result_dict

    def build_lexicon(self, result_dict, threshold = 0.5):

        final_pos_lexicon = dict()
        final_neg_lexicon = dict()
        for word, probs in result_dict.items():
            for index, prob in zip(range(3), probs):
                if prob >= threshold and re.search(r'^[A-Za-z _-]+$', word):
                    if index == 0:
                        final_pos_lexicon[word] = prob
                    elif index == 1:
                        final_neg_lexicon[word] = prob

        print(len(final_pos_lexicon), len(final_neg_lexicon))

        final_pos_lexicon = sorted(final_pos_lexicon.items(), key=lambda item: (item[1], item[0]), reverse=True)
        final_neg_lexicon = sorted(final_neg_lexicon.items(), key=lambda item: (item[1], item[0]), reverse=True)
        pos_enron_path = variables['pos_enron_path']
        f = open(pos_enron_path, mode='w')
        for word, prob in final_pos_lexicon:
            f.writelines(word + ' ' + str(prob) + '\n')
        f.close()
        neg_enron_path = variables['neg_enron_path']
        f = open(neg_enron_path, mode='x')
        for word, prob in final_neg_lexicon:
            f.writelines(word + ' ' + str(prob) + '\n')
        f.close()

if __name__ == '__main__':

    seed_polarity_dict = pickle.load(open('/Volumes/Data/NLP/Automatic_Lexicon_Creation/Enron_Lexicon/Lexicons/seed_polarity_dict.pck','rb'))
    w2v_embedding_path = variables['w2v_embedding_path']
    data_dict = pickle.load(open(w2v_embedding_path, 'rb'))
    X = [data_dict[word] for word in seed_polarity_dict]
    Y = [[1 if y == 1 else 0, 1 if y == -1 else 0, 1 if y == 0 else 0] for word, y in seed_polarity_dict.items()]

    all_X = [embd for word, embd in data_dict.items()]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=1)

    print(len(X_train), len(X_test))
    print('Positive', 'Negative', 'Neutral')
    print('Train set')
    print(np.sum(y_train, axis=0))
    print('Test set')
    print(np.sum(y_test, axis=0))

    sft = Softmax_Lexicon()
    result_dict = sft.train(X_train, X_test, y_train, y_test, all_X,data_dict)
    sft.build_lexicon(result_dict,0.5)
