import unittest
import os
from dotenv import load_dotenv

import nlpaug.augmenter.word as naw
import nlpaug.model.lang_models as nml
from nlpaug.util import Action, Warning


class TestWord(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env_config_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', '.env'))
        load_dotenv(env_config_path)

    def test_empty_input_for_insert(self):
        text = ' '

        augs = [
            naw.BertAug(action=Action.INSERT),
            naw.TfIdfAug(model_path=os.environ.get("MODEL_DIR"), action=Action.SUBSTITUTE)
        ]

        for aug in augs:
            augmented_text = aug.augment(text)
            # FIXME: standardize return
            is_equal = augmented_text == '' or augmented_text == ' '
            self.assertTrue(is_equal)

    def test_empty_input_substitute(self):
        text = ' '
        augs = [
            naw.SpellingAug(dict_path=os.environ.get("MODEL_DIR") + 'spelling_en.txt')
        ]

        for aug in augs:
            augmented_text = aug.augment(text)
            self.assertEqual(text, augmented_text)

    def test_empty_input_for_swap(self):
        texts = [' ']
        aug = naw.RandomWordAug(action=Action.SWAP)
        for text in texts:
            augmented_text = aug.augment(text)

            self.assertEqual(text, augmented_text)

        self.assertEqual(1, len(texts))

        tokens = [None]
        aug = naw.RandomWordAug(action=Action.SWAP)
        for t in tokens:
            augmented_text = aug.augment(t)
            self.assertEqual(augmented_text, None)

        self.assertEqual(len(tokens), 1)

    def test_empty_input_for_delete(self):
        text = ' '
        # None
        augs = [
            naw.RandomWordAug(action=Action.DELETE),
            naw.StopWordsAug(action=Action.DELETE, stopwords=['a', 'an', 'the'])
        ]

        for aug in augs:
            augmented_text = aug.augment(text)
            # FIXME: standardize return
            is_equal = augmented_text == '' or augmented_text == ' '
            self.assertTrue(is_equal)