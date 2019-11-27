import numpy as np

import nlpaug.util.math.normalization as normalization


class WordEmbeddings:
    def __init__(self, top_k=100, skip_check=True):
        self.top_k = top_k
        self.skip_check = skip_check
        self.emb_size = 0
        self.vocab_size = 0
        self.embs = {}
        self.w2v = {}
        self.i2w = {}
        self.w2i = {}
        self.vectors = []
        self.normalized_vectors = None

    def read(self, file_path, max_num_vector):
        raise NotImplementedError

    def similar(self, word):
        raise NotImplementedError

    def download(self, model_path):
        raise NotImplementedError

    def word2idx(self, word):
        return self.w2i[word]

    def word2vector(self, word):
        return self.w2v[word]

    def idx2word(self, idx):
        return self.i2w[idx]

    def get_vectors(self):
        return self.normalized_vectors

    def get_vocab(self):
        return [word for word in self.w2v]

    @classmethod
    def _normalize(cls, vectors, norm='l2'):
        if norm == 'l2':
            return normalization.l2_norm(vectors)
        elif norm == 'l1':
            return normalization.l1_norm(vectors)
        elif norm == 'standard':
            return normalization.standard_norm(vectors)

    def predict(self, word, n=1):
        source_id = self.word2idx(word)
        source_vector = self.word2vector(word)
        scores = np.dot(self.normalized_vectors, source_vector)  # TODO: very slow.
        target_ids = np.argpartition(-scores, self.top_k+2)[:self.top_k+2]  # TODO: slow.
        target_words = [self.idx2word(idx) for idx in target_ids if idx != source_id and self.idx2word(idx).lower() !=
                        word.lower()]  # filter out same word
        return target_words[:self.top_k]
