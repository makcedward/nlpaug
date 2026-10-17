try:
    from gensim.models import KeyedVectors
except ImportError:
    # No installation required if not using this function
    pass

from nlpaug.model.word_embs import WordEmbeddings

pre_trained_model_url = {
    'glove_6b': 'https://nlp.stanford.edu/data/glove.6B.zip',
    'glove_42b_300d': 'https://nlp.stanford.edu/data/glove.42B.300d.zip',
    'glove_840b_300d': 'https://nlp.stanford.edu/data/glove.840B.300d.zip',
    'glove_twitter_27b': 'https://nlp.stanford.edu/data/glove.twitter.27B.zip',
}


class GloVe(WordEmbeddings):
    # https://nlp.stanford.edu/pubs/glove.pdf
    def __init__(self, top_k=100, skip_check=False):
        super().__init__(top_k, skip_check)

        try:
            from gensim.models import KeyedVectors
        except ModuleNotFoundError:
            raise ModuleNotFoundError('Missed gensim library. Install transfomers by `pip install gensim`')

        self.model = None
        self.words = []

    def read(self, file_path, max_num_vector=None):
        import os

        # Issue #269: users hit confusing errors when the GloVe path is wrong
        # or gensim is outdated. Fail fast with actionable messages.
        if not os.path.isfile(file_path):
            raise FileNotFoundError(
                f"GloVe file not found: {file_path!r}. Download it first "
                "(e.g. via nlpaug download utilities) and pass the .txt path "
                "to model_path (issue #269).")
        from gensim.models import KeyedVectors
        try:
            self.model = KeyedVectors.load_word2vec_format(
                file_path, binary=False, no_header=True, limit=max_num_vector)
        except TypeError as e:
            # gensim < 4.0 does not support `no_header` (see issues #269, #322).
            # Fall back to converting header-less GloVe to word2vec format.
            if 'no_header' not in str(e):
                raise
            self.model = self._load_without_no_header(file_path, max_num_vector)
        super()._read()

    @staticmethod
    def _load_without_no_header(file_path, max_num_vector=None):
        from gensim.models import KeyedVectors

        try:
            from gensim.scripts.glove2word2vec import glove2word2vec
        except ImportError:
            glove2word2vec = None

        if glove2word2vec is not None:
            import tempfile
            import os

            fd, tmp_path = tempfile.mkstemp(suffix='.w2v.txt')
            os.close(fd)
            try:
                glove2word2vec(file_path, tmp_path)
                return KeyedVectors.load_word2vec_format(tmp_path, binary=False, limit=max_num_vector)
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        # Minimal fallback when glove2word2vec script is unavailable:
        # prepend "<vocab_size> <dim>" header to a temporary copy.
        import tempfile
        import os

        fd, tmp_path = tempfile.mkstemp(suffix='.w2v.txt')
        try:
            with open(file_path, 'r', encoding='utf-8', errors='strict') as fin, \
                    open(tmp_path, 'w', encoding='utf-8') as fout:
                lines = []
                dim = None
                for line in fin:
                    line = line.rstrip('\n')
                    if not line:
                        continue
                    if max_num_vector is not None and len(lines) >= max_num_vector:
                        break
                    parts = line.split(' ')
                    if dim is None:
                        dim = len(parts) - 1
                    lines.append(line)
                fout.write('{} {}\n'.format(len(lines), dim or 0))
                for line in lines:
                    fout.write(line + '\n')
            return KeyedVectors.load_word2vec_format(tmp_path, binary=False, limit=max_num_vector)
        finally:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                os.remove(tmp_path)
            except OSError:
                pass
