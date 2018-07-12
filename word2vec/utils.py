import os

import numpy as np
import random
from collections import Counter

import tensorflow as tf

def safe_mkdir(path):
    try:
        os.makedirs(path)
    except OSError:
        print('fail', path)
        pass

def read_data(file_path):
    """Read data into a list"""
    with open(file_path,'r') as f:
        words = tf.compat.as_str(f.read()).split()
    return words


def build_vocab(words, vocab_size, sub_sample, visual_fld):
    """ Build vocabulary of VOCAB_SIZE most frequent words and write it to
    visualization/vocab.tsv
    """
    safe_mkdir(visual_fld)
    file = open(os.path.join(visual_fld, 'vocab.tsv'), 'w')

    if sub_sample:
        threshold = 1e-5
        word_counts = Counter(words)
        total_count = len(words)
        freqs = {word: count / total_count for word, count in word_counts.items()}
        p_drop = {word: 1 - np.sqrt(threshold / freqs[word]) for word in word_counts}
        words = [word for word in words if random.random() < (1 - p_drop[word])]

    dictionary = dict()
    count = [('UNK', -1)]  # if word in dataset is not in the dictionary, its index is -1.
    index = 0
    count.extend(Counter(words).most_common(vocab_size - 1))

    for word, _ in count:
        dictionary[word] = index
        index += 1
        file.write(word + '\n')

    index_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    file.close()
    return dictionary, index_dictionary

def convert_words_to_index(words, dictionary):
    """ Replace each word in the dataset with its index in the dictionary """
    return [dictionary[word] if word in dictionary else 0 for word in words]

def generate_sample(index_words, context_window_size):
    """ Form training pairs according to the skip-gram model. """
    for index, center in enumerate(index_words):
        context = random.randint(1, context_window_size)
        # get a random target before the center word
        for target in index_words[max(0, index - context): index]:
            yield center, target
        # get a random target after the center wrod
        for target in index_words[index + 1: index + context + 1]:
            yield center, target

def batch_gen(vocab_size, batch_size,
              skip_window, sub_sample, visual_fld):
    local_dest = 'data/text8'
    words = read_data(local_dest)
    dictionary, _ = build_vocab(words, vocab_size, sub_sample, visual_fld)
    index_words = convert_words_to_index(words, dictionary)
    del words  # to save memory
    single_gen = generate_sample(index_words, skip_window)

    while True:
        center_batch = np.zeros(batch_size, dtype=np.int32)
        target_batch = np.zeros([batch_size,1], dtype=np.int32)
        for index in range(batch_size):
            center_batch[index], target_batch[index] = next(single_gen)
        yield center_batch, target_batch

def most_common_words(visual_fld, num_visualize):
    """ create a list of num_visualize most frequent words to visualize on TensorBoard.
    saved to visualization/vocab_[num_visualize].tsv
    """
    words = open(os.path.join(visual_fld, 'vocab.tsv'), 'r').readlines()[:num_visualize]
    words = [word for word in words]
    file = open(os.path.join(visual_fld, 'vocab_' + str(num_visualize) + '.tsv'), 'w')
    for word in words:
        file.write(word)
    file.close()

if __name__ == '__main__':
    a = read_data('data/text8')
    print(len(a))