import os
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

import nlpaug.model.word_embs.word2vec as w2v_mod


class FakeModel:
    def __init__(self, words):
        self.index_to_key = words
        self.key_to_index = {w: i for i, w in enumerate(words)}
        self._vectors = {w: np.zeros(3) for w in words}

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._vectors[self.index_to_key[key]]
        return self._vectors[key]


def make_fake_gensim(calls, fail_with):
    """fail_with: dict mapping binary flag -> exception to raise (or None)."""
    fake_gensim = types.ModuleType('gensim')
    fake_models = types.ModuleType('gensim.models')

    class FakeKeyedVectors:
        @classmethod
        def load_word2vec_format(cls, *args, **kwargs):
            calls.append(dict(kwargs))
            error = fail_with.get(kwargs.get('binary'))
            if error is not None:
                raise error
            return FakeModel(['king', 'queen'])

    fake_models.KeyedVectors = FakeKeyedVectors
    fake_gensim.models = fake_models
    return fake_gensim, fake_models


class TestWord2vecFormatDetect(unittest.TestCase):
    def _make_file(self, content=b'2 3\nking 0.1 0.2 0.3\n'):
        fd, path = tempfile.mkstemp(suffix='.vec')
        with os.fdopen(fd, 'wb') as f:
            f.write(content)
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
        return path

    def _read(self, path, fail_with, **kwargs):
        calls = []
        fake_gensim, fake_models = make_fake_gensim(calls, fail_with)
        with patch.dict(sys.modules, {'gensim': fake_gensim, 'gensim.models': fake_models}):
            model = w2v_mod.Word2vec()
            model.read(path, **kwargs)
        return calls, model

    def test_binary_file_loads_without_flag(self):
        # Issue #308: binary file must load even though text parsing of the
        # same bytes raises UnicodeDecodeError.
        calls, model = self._read(
            self._make_file(content=b'\x80binary-bytes'),
            fail_with={False: UnicodeDecodeError('utf-8', b'\x80', 0, 1, 'invalid start byte')})
        self.assertEqual(['king', 'queen'], model.get_vocab())
        self.assertEqual(True, calls[0].get('binary'))

    def test_text_file_falls_back_from_binary(self):
        calls, model = self._read(
            self._make_file(),
            fail_with={True: struct.error('unpack requires a buffer')})
        self.assertEqual(['king', 'queen'], model.get_vocab())
        self.assertEqual([True, False], [c.get('binary') for c in calls])

    def test_explicit_flag_still_falls_back_once(self):
        calls, model = self._read(
            self._make_file(),
            fail_with={False: UnicodeDecodeError('utf-8', b'\x80', 0, 1, 'invalid start byte')},
            binary=False)
        self.assertEqual(['king', 'queen'], model.get_vocab())
        self.assertEqual([False, True], [c.get('binary') for c in calls])

    def test_both_formats_failing_raises_helpful_error(self):
        with self.assertRaises(ValueError) as ctx:
            self._read(
                self._make_file(),
                fail_with={True: UnicodeDecodeError('utf-8', b'\x80', 0, 1, 'x'),
                           False: ValueError('invalid file format')})
        self.assertIn('binary', str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
