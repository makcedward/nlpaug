import unittest

import nlpaug.augmenter.word as naw


class TestSplit(unittest.TestCase):
    def test_split(self):
        texts = [
            'The quick brown fox jumps over the lazy dog'
        ]
        aug = naw.SplitAug()

        for text in texts:
            augmented_text = aug.augment(text)

            self.assertLess(len(text), len(augmented_text))

    def test_split_min_char(self):
        texts = [
            'quick brown'
        ]
        aug = naw.SplitAug(min_char=6)

        for text in texts:
            augmented_text = aug.augment(text)
            self.assertEqual(text, augmented_text)
