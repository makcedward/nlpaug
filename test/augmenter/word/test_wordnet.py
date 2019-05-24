import unittest

import nlpaug.augmenter.word as naw


class TestWordNet(unittest.TestCase):
    def test_substitute(self):
        texts = [
            'The quick brown fox jumps over the lazy dog'
        ]
        aug = naw.WordNetAug()

        for text in texts:
            self.assertLess(0, len(text))
            augmented_text = aug.augment(text)

            self.assertNotEqual(text, augmented_text)

        self.assertLess(0, len(texts))

