import numpy as np
from nlpaug.model.word_embs import WordEmbeddings


class Fasttext(WordEmbeddings):
    def __init__(self):
        super(Fasttext, self).__init__()

    def read(self, file_path, max_num_vector=None):
        with open(file_path, 'r', encoding='utf-8') as f:
            header = f.readline()
            self.vocab_size, self.emb_size = map(int, header.split())

            for i, line in enumerate(f):
                tokens = line.split()
                word = " ".join(tokens[0:(len(tokens) - self.emb_size):])
                values = np.array([float(val) for val in tokens[(self.emb_size*-1):]]) 

                self.vectors.append(values)
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
