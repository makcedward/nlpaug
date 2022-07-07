import unittest
import os

import nlpaug.augmenter.sentence as nas


class TestRandomSentenceAug(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = 'This is sentence1. This is sentence2! This is sentence3? This is, sentence4 with comma.'

    def test_mode(self):
        for mode in ['left', 'right', 'neighbor', 'random']:
            aug = nas.RandomSentAug(mode='left')
            aug_data = aug.augment(self.data)
            self.assertNotEqual(self.data, aug_data[0])
            self.assertEqual(4, len(aug.model.tokenize(aug_data[0])))
