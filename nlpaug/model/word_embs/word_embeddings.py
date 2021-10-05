import nlpaug.util.math.normalization as normalization


class WordEmbeddings:
    def __init__(self, top_k=100, skip_check=True):
        self.top_k = top_k
        self.skip_check = skip_check
        self.emb_size = 0
        self.vocab_size = 0
        self.words = []

    def read(self, file_path, max_num_vector):
        raise NotImplementedError

    def _read(self):
        self.words = [self.model.index_to_key[i] for i in range(len(self.model.index_to_key))]
        self.emb_size = self.model[self.model.key_to_index[self.model.index_to_key[0]]]
        self.vocab_size = len(self.words)

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
