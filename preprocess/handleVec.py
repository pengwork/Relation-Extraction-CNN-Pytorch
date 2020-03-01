"""处理词向量文件供下文使用"""

import numpy as np
import json
import pickle


def load_dense(path, vocab, first_line=False):
    words_embed = {}
    dim = 0
    with open(path, "r") as f:
        for line in f:
            if first_line:
                first_line = False
                continue
            line_list = line.split()
            word = line_list[0]
            embed = line_list[1:]
            dim = len(embed)
            embed = np.array([float(num) for num in embed])
            words_embed[word] = embed

    word_to_idx = {word: i + 1 for i, word in enumerate(vocab)}
    word_to_idx['<unk>'] = 0

    idx_to_word = {ix: w for w, ix in word_to_idx.items()}

    id2emb = dict()
    id2emb[0] = np.array([0.0] * dim)

    for idx in range(len(idx_to_word)):
        if idx_to_word[idx] in words_embed:
            id2emb[idx] = words_embed[idx_to_word[idx]]
        else:
            id2emb[idx] = id2emb[0]

    data = np.array([id2emb[ix] for ix in range(len(idx_to_word))])

    return data, word_to_idx


def get_vocab():
    vocab = set()
    with open('../data/train.txt', 'r') as f:
        for line in f:
            line = json.loads(line)
            vocab = vocab | (set(line['sentence'].split(' ')))
    with open('../data/test.txt', 'r') as f:
        for line in f:
            line = json.loads(line)
            vocab = vocab | (set(line['sentence'].split(' ')))
    return vocab


if __name__ == '__main__':
    # path = '../data/vector_50d.txt'
    path = '/Users/libo/Documents/DATA/词向量/glove.6B/glove.6B.300d.txt'

    vocab = get_vocab()

    vocab_data, word_to_idx = load_dense(path, vocab)

    with open('../data/word_embedding_numpy.pkl', 'wb') as f:
        pickle.dump(vocab_data, f)

    with open('../data/word_embedding_word2idx.pkl', 'wb') as f:
        pickle.dump(word_to_idx, f)
