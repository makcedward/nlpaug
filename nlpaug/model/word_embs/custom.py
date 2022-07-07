try:
    from gensim.models import KeyedVectors
except ImportError:
    # No installation required if not using this function
    pass

from nlpaug.model.word_embs import WordEmbeddings


class Custom(WordEmbeddings):
    def __init__(self, top_k=100, skip_check=False):
        super().__init__(top_k, skip_check)

        try:
            from gensim.models import KeyedVectors
        except ModuleNotFoundError:
            raise ModuleNotFoundError('Missed gensim library. Install transfomers by `pip install gensim`')

        self.model = None
        self.words = []

    def read(self, file_path, binary=False, no_header=False, max_num_vector=None):
        self.model = KeyedVectors.load_word2vec_format(file_path, binary=binary, no_header=no_header, limit=max_num_vector)
        super()._read()
