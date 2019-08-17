import unittest

import nlpaug.augmenter.word as naw
from nlpaug.util import Action


class TestRandom(unittest.TestCase):
    def test_swap(self):
        texts = [
            'The quick brown fox jumps over the lazy dog'
        ]
        aug = naw.RandomWordAug(action=Action.SWAP)

        for text in texts:
            self.assertLess(0, len(text))
            augmented_text = aug.augment(text)

            self.assertNotEqual(text, augmented_text)

        self.assertLess(0, len(texts))

    def test_delete(self):
        texts = [
            'The quick brown fox jumps over the lazy dog'
        ]
        aug = naw.RandomWordAug()

        for text in texts:
            self.assertLess(0, len(text))
            augmented_text = aug.augment(text)

            self.assertNotEqual(text, augmented_text)

        self.assertLess(0, len(texts))
