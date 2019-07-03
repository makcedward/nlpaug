import numpy as np
from nlpaug.model.word_embs import WordEmbeddings


class Fasttext(WordEmbeddings):
    def __init__(self, cache=True, skip_check=False):
        super().__init__(cache, skip_check)

    def read(self, file_path, max_num_vector=None):
        with open(file_path, 'r', encoding='utf-8') as f:
            header = f.readline()
            self.vocab_size, self.emb_size = map(int, header.split())

            for i, line in enumerate(f):
                tokens = line.split()
                values = [val for val in tokens[(self.emb_size * -1):]]
                value_pos = line.find(' '.join(values))
                word = line[:value_pos-1]
                values = np.array([float(val) for val in values])

                self.vectors.append(values)
                self.i2w[len(self.i2w)] = word
                self.w2i[word] = len(self.w2i)
                self.w2v[word] = values

        self.vectors = np.asarray(self.vectors)
        if not self.skip_check:
            assert len(self.vectors) == len(self.i2w), \
                'Vector Size:{}, Index2Word Size:{}'.format(len(self.vectors), len(self.i2w))
            assert len(self.i2w) == len(self.w2i), \
                'Index2Word Size:{}, Word2Index Size:{}'.format(len(self.i2w), len(self.w2i))
            assert len(self.w2i) == len(self.w2v), \
                'Word2Index Size:{}, Word2Vector Size:{}'.format(len(self.w2i), len(self.w2v))

        self.normalized_vectors = self._normalize(self.vectors)

        if self.cache:
            self.vocab = [word for word in self.w2v]
