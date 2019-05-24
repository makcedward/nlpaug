import unittest
import os
from dotenv import load_dotenv

import nlpaug.augmenter.word as naw
from nlpaug.util import Action


class TestGloVe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env_config_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', '.env'))
        load_dotenv(env_config_path)

    def test_insert(self):
        texts = [
            'The quick brown fox jumps over the lazy dog'
        ]

        aug = naw.GloVeAug(
            model_path=os.environ.get("MODEL_DIR")+'glove.6B.50d.txt',
            action=Action.INSERT)

        for text in texts:
            tokens = aug.tokenizer(text)
            results = aug.augment(text)

            self.assertLess(len(tokens), len(results))
            self.assertLess(0, len(tokens))

        self.assertLess(0, len(texts))

    def test_substitute(self):
        texts = [
            'The quick brown fox jumps over the lazy dog'
        ]

        aug = naw.GloVeAug(
            model_path=os.environ.get("MODEL_DIR") + 'glove.6B.50d.txt',
            action=Action.SUBSTITUTE)

        for text in texts:
            self.assertLess(0, len(text))
            augmented_text = aug.augment(text)

            self.assertNotEqual(text, augmented_text)

        self.assertLess(0, len(texts))

