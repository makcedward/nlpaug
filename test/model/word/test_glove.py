import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

import nlpaug.model.word_embs.glove as glove_mod
from nlpaug.model.word_embs.word_embeddings import WordEmbeddings


class FakeModel:
    def __init__(self, words):
        self.index_to_key = words
        self.key_to_index = {w: i for i, w in enumerate(words)}
        self._vectors = {w: np.zeros(3) for w in words}

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._vectors[self.index_to_key[key]]
        return self._vectors[key]


def make_fake_gensim(old_gensim, calls):
    fake_gensim = types.ModuleType('gensim')
    fake_models = types.ModuleType('gensim.models')

    class FakeKeyedVectors:
        @classmethod
        def load_word2vec_format(cls, *args, **kwargs):
            calls.append(dict(kwargs))
            if old_gensim and 'no_header' in kwargs:
                raise TypeError("load_word2vec_format() got an unexpected keyword argument 'no_header'")
            return FakeModel(['king', 'queen'])

    fake_models.KeyedVectors = FakeKeyedVectors
    fake_gensim.models = fake_models
    return fake_gensim, fake_models


class TestGloVeNoHeaderCompat(unittest.TestCase):
    def _make_glove_file(self):
        fd, path = tempfile.mkstemp(suffix='.txt')
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write('king 0.1 0.2 0.3\n')
            f.write('queen 0.4 0.5 0.6\n')
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
        return path

    def _run_read(self, old_gensim):
        calls = []
        fake_gensim, fake_models = make_fake_gensim(old_gensim, calls)
        glove_path = self._make_glove_file()
        with patch.dict(sys.modules, {'gensim': fake_gensim, 'gensim.models': fake_models}):
            sys.modules.pop('gensim.scripts.glove2word2vec', None)
            aug = glove_mod.GloVe()
            aug.read(glove_path)
            vocab = aug.get_vocab()
        return calls, vocab

    def test_modern_gensim_uses_no_header(self):
        calls, vocab = self._run_read(old_gensim=False)
        self.assertEqual(vocab, ['king', 'queen'])
        self.assertTrue(any(c.get('no_header') is True for c in calls))

    def test_old_gensim_falls_back_without_no_header(self):
        calls, vocab = self._run_read(old_gensim=True)
        self.assertEqual(vocab, ['king', 'queen'])
        self.assertGreaterEqual(len(calls), 2)
        self.assertIn('no_header', calls[0])
        self.assertNotIn('no_header', calls[-1])


class TestWordEmbeddingsGensimCompat(unittest.TestCase):
    # Regression tests for issues #273/#279:
    # 'Word2VecKeyedVectors' object has no attribute 'index_to_key'
    # (gensim 3.x uses index2word/vocab instead of index_to_key/key_to_index).
    def test_new_gensim_api(self):
        class FakeNewModel:
            def __init__(self):
                self.index_to_key = ['king', 'queen']
                self.key_to_index = {'king': 0, 'queen': 1}
            def __getitem__(self, key):
                return np.zeros(3)

        w = WordEmbeddings()
        w.model = FakeNewModel()
        w._read()
        self.assertEqual(['king', 'queen'], w.get_vocab())
        self.assertEqual(2, w.vocab_size)
        self.assertEqual(3, w.emb_size)

    def test_old_gensim_api(self):
        class FakeOldModel:
            def __init__(self):
                self.index2word = ['king', 'queen']
                self.vocab = {'king': 1, 'queen': 2}
            def __getitem__(self, key):
                return np.zeros(5)

        w = WordEmbeddings()
        w.model = FakeOldModel()
        w._read()
        self.assertEqual(['king', 'queen'], w.get_vocab())
        self.assertEqual(2, w.vocab_size)
        self.assertEqual(5, w.emb_size)


if __name__ == '__main__':
    unittest.main()
