import unittest

from nlpaug.augmenter.word import RandomWordAug
from nlpaug.util import Action


class TestRandom(unittest.TestCase):
    def test_delete(self):
        texts = [
            'The quick brown fox jumps over the lazy dog'
        ]
        aug = RandomWordAug()

        for text in texts:
            tokens = text.split(' ')
            results = aug.augment(tokens)

            self.assertLess(len(results), len(tokens))
            self.assertLess(0, len(tokens))

        self.assertLess(0, len(texts))

