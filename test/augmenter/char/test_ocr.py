import unittest

from nlpaug.augmenter.char import OcrAug


class TestOcr(unittest.TestCase):
    def testSubsituteExistChar(self):
        tokens = ['Zoology', 'roku123456']
        aug = OcrAug()
        for t in tokens:
            augmented_text = aug.augment(t)
            self.assertNotEqual(t, augmented_text)

        self.assertTrue(len(tokens) > 0)

    def testSubsituteNonExistChar(self):
        tokens = ['AAAAA', 'KKKKK']
        aug = OcrAug()
        for t in tokens:
            augmented_text = aug.augment(t)
            self.assertEqual(t, augmented_text)

        self.assertTrue(len(tokens) > 0)