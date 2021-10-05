from gensim.models import KeyedVectors

from nlpaug.model.word_embs import WordEmbeddings


class Fasttext(WordEmbeddings):
    # https://arxiv.org/pdf/1712.09405.pdf,
    def __init__(self, top_k=100, skip_check=False):
        super().__init__(top_k, skip_check)

        self.model = None
        self.words = []

    def read(self, file_path, max_num_vector=None):
        self.model = KeyedVectors.load_word2vec_format(file_path, limit=max_num_vector)
        self.words = [self.model.index_to_key[i] for i in range(len(self.model.index_to_key))]
