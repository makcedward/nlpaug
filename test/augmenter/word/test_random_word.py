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
            augmented_text = aug.augment(text)
            self.assertNotEqual(text, augmented_text)

    def test_substitute_without_target_word(self):
        texts = [
            'The quick brown fox jumps over the lazy dog'
        ]
        aug = naw.RandomWordAug(action='substitute')

        for text in texts:
            augmented_text = aug.augment(text)

            self.assertIn('_', augmented_text)
            self.assertNotEqual(text, augmented_text)

    def test_substitute_with_target_word(self):
        texts = [
            'The quick brown fox jumps over the lazy dog'
        ]
        target_words = ['$', '#', '^^^']
        aug = naw.RandomWordAug(action='substitute', target_words=target_words)

        for text in texts:
            augmented_text = aug.augment(text)

            replaced = False
            for w in target_words:
                if w in augmented_text:
                    replaced = True
                    break
            self.assertTrue(replaced)
            self.assertNotEqual(text, augmented_text)

    def test_delete(self):
        texts = [
            'The quick brown fox jumps over the lazy dog'
        ]
        aug = naw.RandomWordAug()

        for text in texts:
            augmented_text = aug.augment(text)
            self.assertNotEqual(text, augmented_text)
