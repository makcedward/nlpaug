import unittest
import re

import nlpaug.augmenter.char as nac


class TestKeyboard(unittest.TestCase):
    def test_single_word(self):
        texts = ['Zoology', 'roku123456']
        aug = nac.KeyboardAug()
        for text in texts:
            augmented_text = aug.augment(text)
            self.assertNotEqual(text, augmented_text)

        self.assertTrue(len(texts) > 0)

    def test_multi_words(self):
        texts = ['The quick brown fox jumps over the lazy dog']
        aug = nac.KeyboardAug()
        for text in texts:
            augmented_text = aug.augment(text)
            self.assertNotEqual(text, augmented_text)

        self.assertTrue(len(texts) > 0)

    def test_no_special_character(self):
        text = 'qwertyuioplmnbvcxza'
        for i in range(10):
            aug = nac.KeyboardAug(special_char=False)
            augmented_text = aug.augment(text)
            self.assertTrue(re.match("^[a-zA-Z0-9]*$", augmented_text))
