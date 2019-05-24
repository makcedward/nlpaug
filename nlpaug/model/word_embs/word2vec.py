import numpy as np

from nlpaug.model.word_embs import WordEmbeddings


class Word2vec(WordEmbeddings):
    def __init__(self, cache=True):
        super(Word2vec, self).__init__(cache)

    def read(self, file_path, max_num_vector=None):
        with open(file_path, 'rb') as f:
            header = f.readline()
            self.vocab_size, self.emb_size = map(int, header.split())
            if max_num_vector is not None:
                self.vocab_size = min(max_num_vector, self.vocab_size)

            self.vectors = np.zeros((self.vocab_size, self.emb_size), dtype=np.float32)
            binary_len = np.dtype(np.float32).itemsize * self.emb_size

            for _ in range(self.vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == b' ':
                        word = ''.join(word)
                        break
                    if ch != '\n':
                        """print(ch.decode('cp437'))"""
                        word.append(ch.decode('cp437'))
                values = np.fromstring(f.read(binary_len), dtype=np.float32)

                self.vectors[len(self.i2w)] = values
                self.i2w[len(self.i2w)] = word
                self.w2i[word] = len(self.w2i)
                self.w2v[word] = values

        self.vectors = np.asarray(self.vectors)
        assert len(self.vectors) == len(self.i2w)
        assert len(self.i2w) == len(self.w2i)
        assert len(self.w2i) == len(self.w2v)

        self.normalized_vectors = self._normalize(self.vectors)

        if self.cache:
            self.vocab = [word for word in self.w2v]

