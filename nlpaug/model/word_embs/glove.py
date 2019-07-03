import numpy as np

from nlpaug.model.word_embs import WordEmbeddings


class GloVe(WordEmbeddings):
    def __init__(self):
        super().__init__(cache=True, skip_check=False)

    def read(self, file_path, max_num_vector=None):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.split()
                word = tokens[0]
                values = np.array([float(val) for val in tokens[1:]])

                self.vectors.append(values)
                self.i2w[len(self.i2w)] = word
                self.w2i[word] = len(self.w2i)
                self.w2v[word] = values

        self.vectors = np.asarray(self.vectors)
        assert len(self.vectors) == len(self.i2w), \
            'Vector Size:{}, Index2Word Size:{}'.format(len(self.vectors), len(self.i2w))
        assert len(self.i2w) == len(self.w2i), \
            'Index2Word Size:{}, Word2Index Size:{}'.format(len(self.i2w), len(self.w2i))
        assert len(self.w2i) == len(self.w2v), \
            'Word2Index Size:{}, Word2Vector Size:{}'.format(len(self.w2i), len(self.w2v))

        self.normalized_vectors = self._normalize(self.vectors)

        if self.cache:
            self.vocab = [word for word in self.w2v]
