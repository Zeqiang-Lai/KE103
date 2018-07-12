import os

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


# 读取数据
def read_data(path):
    with open(path) as f:
        lines = f.readlines()[2:]
        texts, labels = [], []
        words, tags = [], []
        for line in lines:
            w = line.split()
            if len(w) > 0:
                words.append(w[0])
                tags.append(w[-1])
            else:
                texts.append(words)
                labels.append(tags)
                words = []
                tags = []

        texts.append(words)
        labels.append(tags)

        print('Found {0} sentence'.format(len(texts)))

        return texts, labels


def build_dict(texts, labels):
    # 构建tag的字典
    words_index = {'<pad>': 0}
    labels_index = {'<pad>': 0}
    id = 1
    tags = set()
    for line in labels:
        for tag in line:
            tags.add(tag)
    for tag in tags:
        labels_index[tag] = id
        id += 1

    print('Found {0} unique tag'.format(len(labels_index)))

    id = 1
    words = set()
    for line in texts:
        for word in line:
            words.add(word)
    for word in words:
        words_index[word] = id
        id += 1

    words_index['<unk>'] = id
    print('Found {0} unique word'.format(len(words_index)))

    index_words = dict(zip(words_index.values(), words_index.keys()))
    index_labels = dict(zip(labels_index.values(), labels_index.keys()))

    return words_index, index_words, labels_index, index_labels


def convert_word_to_index(texts, dictionary):
    new_texts = []
    for line in texts:
        new_texts.append([dictionary[word] if word in dictionary else dictionary['<unk>'] for word in line])
    return new_texts


def load_data(data_fld):
    train_path = os.path.join(data_fld, 'train.txt')
    valid_path = os.path.join(data_fld, 'valid.txt')
    test_path = os.path.join(data_fld, 'test.txt')

    texts, labels = read_data(train_path)
    word2idx, idx2word, label2idx, idx2label = build_dict(texts, labels)

    x_train = convert_word_to_index(texts, word2idx)
    y_train = convert_word_to_index(labels, label2idx)

    texts, labels = read_data(valid_path)
    x_valid = convert_word_to_index(texts, word2idx)
    y_valid = convert_word_to_index(labels, label2idx)

    texts, labels = read_data(test_path)
    x_test = convert_word_to_index(texts, word2idx)
    y_test = convert_word_to_index(labels, label2idx)

    print('x_train: {0} y_train: {0}'.format(len(x_train), len(y_train)))
    print('x_valid: {0} y_valid: {0}'.format(len(x_valid), len(y_valid)))
    print('x_test: {0} y_test: {0}'.format(len(x_test), len(y_test)))

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test), (word2idx, idx2word, label2idx, idx2label)


def load_embedding_matrix(glove_path, word_index, EMBEDDING_DIM, MAX_NUM_WORDS):
    embeddings_index = {}
    with open(glove_path, 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))

    num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    print(np.shape(embedding_matrix))
    return embedding_matrix


def process_data_for_keras(label_size, X, y=None):
    word_ids = pad_sequences(X, padding='post')
    lengths = np.array([len(sent) for sent in X], dtype='int32')
    features = [word_ids, lengths]
    # features = [word_ids]
    if y is not None:
        y = pad_sequences(y, padding='post')
        y = to_categorical(y, label_size).astype(int)
        y = y if len(y.shape) == 3 else np.expand_dims(y, axis=0)
        return features, y
    else:
        return features


def batch_iter(data, labels, label_size, batch_size=1, shuffle=True):
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1

    def data_generator():
        """
        Generates a batch iterator for a dataset.
        """
        data_size = len(data)
        while True:
            indices = np.arange(data_size)
            # Shuffle the data at each epoch
            if shuffle:
                indices = np.random.permutation(indices)

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                X = [data[i] for i in indices[start_index: end_index]]
                y = [labels[i] for i in indices[start_index: end_index]]
                yield process_data_for_keras(label_size, X, y)

    return num_batches_per_epoch, data_generator()


if __name__ == '__main__':
    data_fld = 'data'
    train, valid, test, dic = load_data(data_fld)
    x_train, y_train = train
    x_valid, y_valid = valid
    x_test, y_test = test
    word2idx, idx2word, label2idx, idx2label = dic
