import numpy as np

from nlpaug.model.word_embs import WordEmbeddings


class Word2vec(WordEmbeddings):
    def __init__(self, cache=True, skip_check=False):
        super().__init__(cache, skip_check)

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
                        word.append(ch.decode('cp437'))
                values = np.fromstring(f.read(binary_len), dtype=np.float32)

                # word = " ".join(tokens[0:(len(tokens) - self.emb_size):])
                # values = np.array([float(val) for val in tokens[(self.emb_size * -1):]])

                self.vectors[len(self.i2w)] = values
                self.i2w[len(self.i2w)] = word
                self.w2i[word] = len(self.w2i)
                self.w2v[word] = values

        self.vectors = np.asarray(self.vectors)
        if not self.skip_check:
            if len(self.vectors) != len(self.i2w):
                raise AssertionError('Vector Size:{}, Index2Word Size:{}'.format(len(self.vectors), len(self.i2w)))
            if len(self.i2w) != len(self.w2i):
                raise AssertionError('Index2Word Size:{}, Word2Index Size:{}'.format(len(self.i2w), len(self.w2i)))
            if len(self.w2i) != len(self.w2v):
                raise AssertionError('Word2Index Size:{}, Word2Vector Size:{}'.format(len(self.w2i), len(self.w2v)))

        self.normalized_vectors = self._normalize(self.vectors)

        if self.cache:
            self.vocab = [word for word in self.w2v]

