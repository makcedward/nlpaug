import numpy as np
from nlpaug.model.word_embs import WordEmbeddings


class Fasttext(WordEmbeddings):
    # https://arxiv.org/pdf/1712.09405.pdf,
    def __init__(self, top_k=100, skip_check=False):
        super().__init__(top_k, skip_check)

    def read(self, file_path, max_num_vector=None):
        vectors = []
        with open(file_path, 'r', encoding='utf-8') as f:
            header = f.readline()
            self.vocab_size, self.emb_size = map(int, header.split())

            for line in f:
                tokens = line.split()
                values = [val for val in tokens[(self.emb_size * -1):]]
                value_pos = line.find(' '.join(values))
                word = line[:value_pos-1]
                values = np.array([float(val) for val in values])

                vectors.append(values)
                self.i2w[len(self.i2w)] = word
                self.w2i[word] = len(self.w2i)
                self.w2v[word] = values

        vectors = np.asarray(vectors)
        if not self.skip_check:
            if len(vectors) != len(self.i2w):
                raise AssertionError('Vector Size:{}, Index2Word Size:{}'.format(len(vectors), len(self.i2w)))
            if len(self.i2w) != len(self.w2i):
                raise AssertionError('Index2Word Size:{}, Word2Index Size:{}'.format(len(self.i2w), len(self.w2i)))
            if len(self.w2i) != len(self.w2v):
                raise AssertionError('Word2Index Size:{}, Word2Vector Size:{}'.format(len(self.w2i), len(self.w2v)))

        self.normalized_vectors = self._normalize(vectors)
