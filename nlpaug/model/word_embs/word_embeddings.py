import numpy as np

import nlpaug.util.math.normalization as normalization


class WordEmbeddings:
    def __init__(self, cache=True, skip_check=True):
        self.cache = cache
        self.skip_check = skip_check
        self.emb_size = 0
        self.vocab_size = 0
        self.embs = {}
        self.w2v = {}
        self.i2w = {}
        self.w2i = {}
        self.vectors = []
        self.normalized_vectors = None

        self.vocab = []

    def read(self, file_path, max_num_vector):
        raise NotImplemented()

    def similar(self, word):
        raise NotImplemented()

    def word2idx(self, word):
        return self.w2i[word]

    def word2vector(self, word):
        return self.w2v[word]

    def idx2word(self, idx):
        return self.i2w[idx]

    def get_vectors(self, normalize=False):
        if normalize:
            return self.normalized_vectors
        return self.vectors

    def get_vocab(self):
        if self.cache:
            return self.vocab
        return [word for word in self.w2v]

    def _normalize(self, vectors, norm='l2'):
        if norm == 'l2':
            return normalization.l2_norm(vectors)
        elif norm == 'l1':
            return normalization.l1_norm(vectors)
        elif norm == 'standard':
            return normalization.standard_norm(vectors)

    def predict(self, word, top_n=10):
        source_id = self.word2idx(word)
        source_vector = self.word2vector(word)
        scores = np.dot(self.normalized_vectors, source_vector)
        target_ids = np.argpartition(-scores, top_n+1)[:top_n+1]
        target_words = [self.idx2word(idx) for idx in target_ids if idx != source_id]
        return target_words[:top_n]
