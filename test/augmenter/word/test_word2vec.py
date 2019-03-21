import unittest
from os.path import join, dirname
import os
from dotenv import load_dotenv

from nlpaug.augmenter.word import Word2vecAug
from nlpaug.util import Action


class TestWord2vec(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env_config_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', '.env'))
        load_dotenv(env_config_path)

    def test_insert(self):
        texts = [
            'The quick brown fox jumps over the lazy dog'
        ]

        aug = Word2vecAug(
            model_path=os.environ.get("MODEL_DIR")+'GoogleNews-vectors-negative300.bin',
            action=Action.INSERT)

        for text in texts:
            tokens = text.split(' ')
            results = aug.augment(tokens)

            self.assertLess(len(tokens), len(results))
            self.assertLess(0, len(tokens))

        self.assertLess(0, len(texts))

    def test_substitute(self):
        texts = [
            'The quick brown fox jumps over the lazy dog'
        ]

        aug = Word2vecAug(
            model_path=os.environ.get("MODEL_DIR")+'GoogleNews-vectors-negative300.bin',
            action=Action.SUBSTITUTE)

        for text in texts:
            tokens = text.split(' ')
            results = aug.augment(tokens)

            at_least_one_not_equal = False
            for t, r in zip(tokens, results):
                if t != r:
                    at_least_one_not_equal = True
                    break

            self.assertEqual(len(tokens), len(results))
            self.assertTrue(at_least_one_not_equal)
            self.assertLess(0, len(tokens))

        self.assertLess(0, len(texts))

