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

    def download(self, model_path):
        raise NotImplementedError

    def get_vocab(self):
        return self.words

    @classmethod
    def _normalize(cls, vectors, norm='l2'):
        if norm == 'l2':
            return normalization.l2_norm(vectors)
        elif norm == 'l1':
            return normalization.l1_norm(vectors)
        elif norm == 'standard':
            return normalization.standard_norm(vectors)

    def predict(self, word, n=1):
        result = self.model.most_similar(word, topn=self.top_k+1)
        result = [w for w, s in result if w.lower() != word.lower()]
        return result[:self.top_k]
