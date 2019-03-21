import unittest

from nlpaug.augmenter.word import SynonymAug


class TestSynonym(unittest.TestCase):
    def test_substitute(self):
        texts = [
            'The quick brown fox jumps over the lazy dog'
        ]
        aug = SynonymAug()

        for text in texts:
            tokens = text.split(' ')
            results = aug.augment(tokens)

            at_least_one_not_equal = False
            for t, r in zip(tokens, results):
                if t != r:
                    at_least_one_not_equal = True
                    break

            self.assertTrue(at_least_one_not_equal)
            self.assertLess(0, len(tokens))

        self.assertLess(0, len(texts))

