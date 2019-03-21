import unittest

from nlpaug.augmenter.char import OcrAug


class TestOcr(unittest.TestCase):
    def testSubsituteExistChar(self):
        tokens = ['Zoology', 'roku123456']
        aug = OcrAug()
        for t in tokens:
            result = aug.augment([t])[0]
            self.assertNotEqual(t, result)

        self.assertTrue(len(tokens) > 0)

    def testSubsituteNonExistChar(self):
        tokens = ['AAAAA', 'KKKKK']
        aug = OcrAug()
        for t in tokens:
            result = aug.augment([t])[0]
            self.assertEqual(t, result)

        self.assertTrue(len(tokens) > 0)