import numpy as np
import re
import itertools
from collections import Counter
"""
Original taken from https://github.com/dennybritz/cnn-text-classification-tf
"""

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(data):
    """
    Returns split sentences and labels.
    """
    # Load data from files
    # positive_examples = list(open("/Users/gautamborgohain/Google Drive/NTU/Sem 1/Other materials/H6751/ Labs/Lab9/dataset/sentence-level-movie-review/rt-polaritydata/rt-polaritydata/rt-polarity.pos",'rb').readlines())
    # positive_examples = [s.strip() for s in positive_examples]
    # negative_examples = list(open("/Users/gautamborgohain/Google Drive/NTU/Sem 1/Other materials/H6751/ Labs/Lab9/dataset/sentence-level-movie-review/rt-polaritydata/rt-polaritydata/rt-polarity.neg",'rb').readlines())
    # negative_examples = [s.strip() for s in negative_examples]
    # # Split by words
    # x_text = positive_examples + negative_examples
    # x_text = [clean_str(sent.decode('ascii','ignore')) for sent in x_text]
    # x_text = [s.split(" ") for s in x_text]
    # # Generate labels
    # positive_labels = [[0, 1] for _ in positive_examples]
    # negative_labels = [[1, 0] for _ in negative_examples]
    x_text = [ [word for word in sen.split()] for sen in data.Sentences]
    y = [[1 if pol == -1 else 0, 1 if pol == 0 else 0, 1 if pol == 1 else 0] for pol in data.Sentiment]
    return [x_text, y]


def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def load_data(data):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels(data)
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]


def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def save_Keras_model(model,path):
    # serialize model to JSON
    model_json = model.to_json()
    with open(path + '.json', "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(path+'.h5')
    print("Saved model to disk")

def load_Keras_model(path):
    from keras.models import model_from_json
    # load json and create model
    json_file = open(path + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(path + ".h5")
    print("Loaded model from disk")
    return loaded_model