import unittest

import nlpaug.augmenter.char as nac


class TestKeyboard(unittest.TestCase):
    def test_qwerty_single_word(self):
        texts = ['Zoology', 'roku123456']
        aug = nac.KeyboardAug()
        for text in texts:
            augmented_text = aug.augment(text)
            self.assertNotEqual(text, augmented_text)

        self.assertTrue(len(texts) > 0)

    def test_qwerty_multi_words(self):
        texts = ['The quick brown fox jumps over the lazy dog']
        aug = nac.KeyboardAug()
        for text in texts:
            augmented_text = aug.augment(text)
            self.assertNotEqual(text, augmented_text)

        self.assertTrue(len(texts) > 0)
