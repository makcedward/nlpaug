import unittest

from nlpaug.augmenter.char import QwertyAug


class TestOcr(unittest.TestCase):
    def testExistChar(self):
        tokens = ['Zoology', 'roku123456']
        aug = QwertyAug()
        for t in tokens:
            augmented_text = aug.augment(t)
            self.assertNotEqual(t, augmented_text)

        self.assertTrue(len(tokens) > 0)