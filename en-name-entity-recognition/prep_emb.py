import json

import numpy as np

def load_my_emb(word_index, EMBEDDING_DIM, MAX_NUM_WORDS):
    embeddings_index = {}
    emb_path = 'embedding/embedding.txt'
    vocab_path = 'embedding/dict.txt'

    emb = np.loadtxt(emb_path)
    vocab = json.load(open(vocab_path))

    for word, idx in vocab.items():
        embeddings_index[word] = emb[idx]

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
