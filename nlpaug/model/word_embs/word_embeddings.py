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
        # Support both gensim 4.x (index_to_key/key_to_index) and gensim 3.x
        # (index2word/vocab). See issues #273 and #279 where old gensim raises
        # AttributeError: 'Word2VecKeyedVectors' object has no attribute 'index_to_key'.
        model = self.model
        if hasattr(model, 'index_to_key'):
            keys = list(model.index_to_key)
        elif hasattr(model, 'index2word'):
            keys = list(model.index2word)
        elif hasattr(model, 'index2entity'):
            keys = list(model.index2entity)
        else:
            raise AttributeError(
                'Unsupported gensim KeyedVectors API: missing index_to_key/index2word. '
                'See issues #273/#279.')
        self.words = keys
        self.vocab_size = len(self.words)
        if self.words:
            first_word = self.words[0]
            try:
                vec = model[first_word]
            except Exception:
                # gensim 4.x also supports integer lookup via key_to_index
                if hasattr(model, 'key_to_index'):
                    vec = model[model.key_to_index[first_word]]
                else:
                    raise
            try:
                self.emb_size = len(vec)
            except TypeError:
                self.emb_size = vec

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
