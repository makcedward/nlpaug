import numpy as np

from nlpaug.model.word_embs import WordEmbeddings

pre_trained_model_url = {
    'glove_6b': 'http://nlp.stanford.edu/data/glove.6B.zip',
    'glove_42b_300d': 'http://nlp.stanford.edu/data/glove.42B.300d.zip',
    'glove_840b_300d': 'http://nlp.stanford.edu/data/glove.840B.300d.zip',
    'glove_twitter_27b': 'http://nlp.stanford.edu/data/glove.twitter.27B.zip',
}


class GloVe(WordEmbeddings):
    # https://nlp.stanford.edu/pubs/glove.pdf
    def __init__(self, top_k=100, skip_check=False):
        super().__init__(top_k, skip_check)

    def read(self, file_path, max_num_vector=None):
        vectors = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.split()
                token_len = len(tokens) % 25

                # Handle if token length is longer than 1 (e.g. . . . in glove.840B.300d)
                values = np.array([float(val) for val in tokens[token_len:]])

                # Exist two words while one word has extra space (e.g. "pp." and "pp. " in glove.840B.300d)
                word = line[:line.find(str(values[0])) - 1]

                # Skip special word
                if '�' in word:
                    continue

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
