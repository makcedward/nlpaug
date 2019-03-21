import unittest

from nlpaug.augmenter.char import QwertyAug


class TestOcr(unittest.TestCase):
    def testExistChar(self):
        tokens = ['Zoology', 'roku123456']
        aug = QwertyAug()
        for t in tokens:
            result = aug.augment([t])[0]
            self.assertNotEqual(t, result)

        self.assertTrue(len(tokens) > 0)