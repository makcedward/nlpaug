import numpy as np

from nlpaug.model.word_embs import WordEmbeddings
from nlpaug.util.file.download import DownloadUtil

pre_trained_model_url = {
    'glove_6b': 'http://nlp.stanford.edu/data/glove.6B.zip',
    'glove_42b_300d': 'http://nlp.stanford.edu/data/glove.42B.300d.zip',
    'glove_840b_300d': 'http://nlp.stanford.edu/data/glove.840B.300d.zip',
    'glove_twitter_27b': 'http://nlp.stanford.edu/data/glove.twitter.27B.zip',
}


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
