import unittest

from nlpaug.augmenter.char import OcrAug


class TestOcr(unittest.TestCase):
    def test_empty_input(self):
        # Empty input
        tokens = ['']
        aug = OcrAug()
        for t in tokens:
            augmented_text = aug.augment(t)
            self.assertEqual(augmented_text, '')

        self.assertEqual(len(tokens[0]), 0)
        self.assertTrue(len(tokens) > 0)

        tokens = [None]
        aug = OcrAug()
        for t in tokens:
            augmented_text = aug.augment(t)
            self.assertEqual(augmented_text, None)

        self.assertEqual(len(tokens), 1)

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