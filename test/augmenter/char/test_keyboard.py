import unittest
import re
import json
import os

import nlpaug.augmenter.char as nac


class TestKeyboard(unittest.TestCase):
    def test_single_word(self):
        texts = ['Zoology', 'roku123456']
        aug = nac.KeyboardAug()
        for text in texts:
            augmented_data = aug.augment(text)
            augmented_text = augmented_data[0]
            self.assertNotEqual(text, augmented_text)

        self.assertTrue(len(texts) > 0)

    def test_multi_words(self):
        texts = ['The quick brown fox jumps over the lazy dog']
        aug = nac.KeyboardAug()
        for text in texts:
            augmented_data = aug.augment(text)
            augmented_text = augmented_data[0]
            self.assertNotEqual(text, augmented_text)

        self.assertTrue(len(texts) > 0)

    def test_no_special_character(self):
        text = 'qwertyuioplmnbvcxza'
        for i in range(10):
            aug = nac.KeyboardAug(include_special_char=False)
            augmented_data = aug.augment(text)
            augmented_text = augmented_data[0]
            self.assertTrue(re.match("^[a-zA-Z0-9]*$", augmented_text))

    def test_lang_de(self):
        text = 'llllllllllllllllll lllllll'
        aug = nac.KeyboardAug(lang='de')

        augmented = False
        # make sure it convert to at least one of the DE char
        for _ in range(10):
            augmented_data = aug.augment(text)
            augmented_text = augmented_data[0]
            if 'ö' in augmented_text or 'Ö' in augmented_text :
                augmented = True
                self.assertNotEqual(text, augmented_text)

        self.assertTrue(augmented)

    def test_lang_es(self):
        text = 'llllllllllllllllll lllllll'
        aug = nac.KeyboardAug(lang='es')

        augmented = False
        # make sure it convert to at least one of the DE char
        for _ in range(10):
            augmented_data = aug.augment(text)
            augmented_text = augmented_data[0]
            if 'ñ' in augmented_text or 'Ñ' in augmented_text :
                augmented = True
                self.assertNotEqual(text, augmented_text)

        self.assertTrue(augmented)

    def test_lang_fr(self):
        text = 'ççççççççççç ççççççççç'
        aug = nac.KeyboardAug(lang='fr')

        augmented = False
        # make sure it convert to at least one of the DE char
        for _ in range(10):
            augmented_data = aug.augment(text)
            augmented_text = augmented_data[0]
            if 'à' in augmented_text or 'à' in augmented_text :
                augmented = True
                self.assertNotEqual(text, augmented_text)

        self.assertTrue(augmented)

    def test_lang_he(self):
        text = 'את המערכה בתנופה'
        aug = nac.KeyboardAug(lang='he')
        augmented_data = aug.augment(text)
        augmented_text = augmented_data[0]
        self.assertNotEqual(text, augmented_text)

    def test_lang_it(self):
        text = 'llllllllllllllllll lllllll'
        aug = nac.KeyboardAug(lang='it')

        augmented = False
        # make sure it convert to at least one of the DE char
        for _ in range(10):
            augmented_data = aug.augment(text)
            augmented_text = augmented_data[0]
            if 'ò' in augmented_text or 'ç' in augmented_text :
                augmented = True
                self.assertNotEqual(text, augmented_text)

        self.assertTrue(augmented)

    def test_lang_nl(self):
        text = 'jjjjjjjjjjjjjjjjjjjjjjjjj jjjjjjjj'
        aug = nac.KeyboardAug(lang='nl')
        augmented_data = aug.augment(text)
        augmented_text = augmented_data[0]
        self.assertNotEqual(text, augmented_text)

    def test_lang_pl(self):
        text = 'kkkkkkkkkkkkkk kkkkkkkkk'
        aug = nac.KeyboardAug(lang='pl')

        augmented = False
        # make sure it convert to at least one of the DE char
        for _ in range(10):
            augmented_data = aug.augment(text)
            augmented_text = augmented_data[0]
            if 'ń' in augmented_text or 'ś' in augmented_text :
                augmented = True
                self.assertNotEqual(text, augmented_text)

        self.assertTrue(augmented)

    def test_lang_th(self):
        text = 'ฤฤฤฤ ฤฏณ'
        aug = nac.KeyboardAug(lang='th')
        augmented_data = aug.augment(text)
        augmented_text = augmented_data[0]
        self.assertNotEqual(text, augmented_text)

    def test_lang_uk(self):
        text = 'планувалося провести'
        aug = nac.KeyboardAug(lang='uk')
        augmented_data = aug.augment(text)
        augmented_text = augmented_data[0]
        self.assertNotEqual(text, augmented_text)

    def test_lang_tr(self):
        text = 'çığırtkan'
        aug = nac.KeyboardAug(lang='tr')
        augmented_data = aug.augment(text)
        augmented_text = augmented_data[0]
        self.assertNotEqual(text, augmented_text)

    def test_non_support_lang(self):
        try:
            nac.KeyboardAug(lang='non_exist')
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_custom_model(self):
        custom_model = {
            'a': '1',
            'b': '2',
        }

        custom_model_file_path = 'char_keyboard_custom_model.json'

        with open(custom_model_file_path, 'w') as outfile:
            json.dump(custom_model, outfile)

        text = 'ababab'
        aug = nac.KeyboardAug(model_path=custom_model_file_path)
        augmented_data = aug.augment(text)
        augmented_text = augmented_data[0]

        self.assertTrue('1' in augmented_text or '2' in augmented_text)

        if os.path.exists(custom_model_file_path):
            os.remove(custom_model_file_path)

    def test_load_custom_model_fail(self):
        try:
            aug = nac.KeyboardAug(model_path='test_load_custom_model_fail.json')
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)
