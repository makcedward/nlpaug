try:
    from gensim.models import KeyedVectors
except ImportError:
    # No installation required if not using this function
    pass

from nlpaug.model.word_embs import WordEmbeddings


class Word2vec(WordEmbeddings):
    # https://arxiv.org/pdf/1301.3781.pdf
    def __init__(self, top_k=100, skip_check=False):
        super().__init__(top_k, skip_check)

        try:
            from gensim.models import KeyedVectors
        except ModuleNotFoundError:
            raise ModuleNotFoundError('Missed gensim library. Install transfomers by `pip install gensim`')

        self.model = None
        self.words = []

    def read(self, file_path, max_num_vector=None, binary=None):
        """Load a word2vec file, auto-detecting binary vs text format.

        :param str file_path: Path to the vectors file.
        :param int max_num_vector: Maximum number of vectors to load.
        :param bool binary: Default None auto-detects: binary format is tried
            first, then plain-text format (issue #308). Pass True/False to
            force one format; the other format is still retried once if the
            forced format fails with a decoding/parsing error.
        """
        import struct

        from gensim.models import KeyedVectors

        tried = [binary] if binary is not None else [True, False]
        if binary is not None:
            tried.append(not binary)

        last_error = None
        for mode in tried:
            try:
                self.model = KeyedVectors.load_word2vec_format(
                    file_path, binary=mode, limit=max_num_vector)
                super()._read()
                return
            except (UnicodeDecodeError, UnicodeError, ValueError, struct.error, EOFError) as e:
                last_error = e

        raise ValueError(
            "Failed to load word2vec model from '{}' as {} format. "
            "Binary (.bin) files require binary=True while plain-text (.txt/.vec) "
            "files require binary=False.".format(
                file_path, 'binary' if tried[0] else 'plain-text')
        ) from last_error
