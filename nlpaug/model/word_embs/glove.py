from gensim.models import KeyedVectors

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

        self.model = None
        self.words = []

    def read(self, file_path, max_num_vector=None):
        self.model = KeyedVectors.load_word2vec_format(file_path, binary=False, no_header=True, limit=max_num_vector)
        self.words = [self.model.index_to_key[i] for i in range(len(self.model.index_to_key))]