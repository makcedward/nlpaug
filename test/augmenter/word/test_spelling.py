import unittest
import os
from dotenv import load_dotenv

import nlpaug.augmenter.word as naw


class TestSpelling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env_config_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', '.env'))
        load_dotenv(env_config_path)

        cls.model_dir = os.path.join(os.environ.get("PACKAGE_DIR"), 'res', 'word', 'spelling')

    def test_read_default_dict(self):
        text = 'abcdef'

        aug = naw.SpellingAug()
        self.assertTrue(aug.model.dict_path)
        aug.augment(text)
        self.assertTrue(True)

    def test_oov(self):
        text = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'

        aug = naw.SpellingAug(dict_path=os.path.join(self.model_dir, 'spelling_en.txt'))
        augmented_data = aug.augment(text)
        augmented_text = augmented_data[0]

        self.assertEqual(text, augmented_text)

    def test_substitute(self):
        texts = [
            'The quick brown fox jumps over the lazy dog'
        ]

        aug = naw.SpellingAug(dict_path=os.path.join(self.model_dir, 'spelling_en.txt'))

        for text in texts:
            self.assertLess(0, len(text))
            augmented_data = aug.augment(text)
            augmented_text = augmented_data[0]

            self.assertNotEqual(text, augmented_text)

        self.assertLess(0, len(texts))

    def test_substitute_stopwords(self):
        texts = [
            'The quick brown fox jumps over the lazy dog'
        ]

        stopwords = [t.lower() for t in texts[0].split(' ')[:3]]
        aug_n = 3

        aug = naw.SpellingAug(dict_path=os.path.join(self.model_dir, 'spelling_en.txt'), stopwords=stopwords)

        for text in texts:
            self.assertLess(0, len(text))
            augmented_data = aug.augment(text)
            augmented_text = augmented_data[0]

            augmented_tokens = aug.tokenizer(augmented_text)
            tokens = aug.tokenizer(text)

            augmented_cnt = 0

            for token, augmented_token in zip(tokens, augmented_tokens):
                if token.lower() in stopwords and len(token) > aug_n:
                    self.assertEqual(token.lower(), augmented_token)
                else:
                    augmented_cnt += 1

            self.assertGreater(augmented_cnt, 0)

        self.assertLess(0, len(texts))
