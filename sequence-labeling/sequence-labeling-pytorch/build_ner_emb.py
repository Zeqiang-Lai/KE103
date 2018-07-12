import argparse
import os

import numpy as np

import utils

parser = argparse.ArgumentParser()
parser.add_argument('--emb_dir', default='embedding', help="Directory containing the word embedding")
parser.add_argument('--out_dir', default='embedding/ner', help="Directory containing the processed word embedding")
parser.add_argument('--words_dir', default='data/ner')
parser.add_argument('--gen_dim', default=300)
parser.add_argument('--domain_dim', default=100)
parser.add_argument('--max_num_words', default=30000)

def load_embedding_matrix(path_emb, word_index, emb_dim, max_num_words):
    embeddings_index = {}
    with open(path_emb, 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    num_words = min(max_num_words, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, emb_dim))
    for word, i in word_index.items():
        if i >= max_num_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


def load_word_idx(path_words):
    with open(path_words, 'r') as f:
        word_idx = {}
        for i, word in enumerate(f.readlines()):
            word = word.strip().split()[0]
            word_idx[word] = i
        return word_idx

if __name__ == '__main__':
    args = parser.parse_args()

    word_idx = load_word_idx('data/ner/words.txt')
    path_emb_glove = os.path.join(args.emb_dir, 'glove.6B.300d.txt')
    gen_emb = load_embedding_matrix(path_emb_glove, word_idx, args.gen_dim, args.max_num_words)

    path_out_emb = path_out_emb_restaurants = os.path.join(args.emb_dir, 'ner')
    if not os.path.exists(path_out_emb):
        os.makedirs(path_out_emb)
    np.save(os.path.join(path_out_emb, 'gen.npy'), gen_emb)

    # Save embeddings properties in json file
    sizes = {
        'gen_embedding_dim': args.gen_dim,
        'domain_embedding_dim': args.domain_dim,
    }
    utils.save_dict_to_json(sizes, os.path.join(args.out_dir, 'embedding_params.json'))

    # Logging sizes
    to_print = "\n".join("- {}: {}".format(k, v) for k, v in sizes.items())
    print("Characteristics of the embedding:\n{}".format(to_print))