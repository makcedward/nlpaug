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

    def test_lang_th(self):
        text = 'ฤฤฤฤ ฤฏณ'
        aug = nac.KeyboardAug(lang='th')
        augmented_text = aug.augment(text)
        self.assertNotEqual(text, augmented_text)

    def test_non_support_lang(self):
        try:
            nac.KeyboardAug(lang='non_exist')
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)
