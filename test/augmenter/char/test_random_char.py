import unittest

from nlpaug.augmenter.char.random import RandomCharAug
from nlpaug.util import Action


class TestRandomCharReplaceAug(unittest.TestCase):
    def testInsertExistChar(self):
        tokens = ['Zoology', 'roku123456']
        aug = RandomCharAug(action=Action.INSERT)
        for t in tokens:
            augmented_text = aug.augment(t)
            self.assertNotEqual(t, augmented_text)
            self.assertLess(len(t), len(augmented_text))

        self.assertTrue(len(tokens) > 0)

    def testSubstituteExistChar(self):
        tokens = ['Zoology', 'roku123456']
        aug = RandomCharAug(action=Action.SUBSTITUTE)
        for t in tokens:
            augmented_text = aug.augment(t)
            self.assertNotEqual(t, augmented_text)

        self.assertTrue(len(tokens) > 0)

    def testSwapChar(self):
        tokens = ['Zoology', 'roku123456']
        aug = RandomCharAug(action=Action.SWAP)
        for t in tokens:
            augmented_text = aug.augment(t)
            self.assertNotEqual(t, augmented_text)

        self.assertTrue(len(tokens) > 0)

    def testSwapStopwords(self):
        tokens = ['Zoology', 'roku123456']
        stopwords = tokens[:1]
        aug = RandomCharAug(action=Action.SWAP, stopwords=stopwords)
        for t in tokens:
            augmented_text = aug.augment(t)
            if t in stopwords:
                self.assertEqual(t, augmented_text)
            else:
                self.assertNotEqual(t, augmented_text)

        self.assertTrue(len(tokens) > 0)

    def testDeleteExistChar(self):
        tokens = ['Zoology', 'roku123456']
        aug = RandomCharAug(action=Action.DELETE)
        for t in tokens:
            augmented_text = aug.augment(t)
            self.assertNotEqual(t, augmented_text)
            self.assertLess(len(augmented_text), len(t))

        self.assertTrue(len(tokens) > 0)

