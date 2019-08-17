import unittest
import os
from dotenv import load_dotenv

import nlpaug.augmenter.word as naw
from nlpaug.util import Action

class TestWordEmbsAug(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env_config_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', '.env'))
        load_dotenv(env_config_path)

        cls.insert_augmenters = [
            naw.Word2vecAug(
                model_path=os.environ.get("MODEL_DIR") + 'GoogleNews-vectors-negative300.bin',
                action=Action.INSERT),
            naw.FasttextAug(
                model_path=os.environ.get("MODEL_DIR") + 'wiki-news-300d-1M.vec',
                action=Action.INSERT),
            naw.GloVeAug(
                model_path=os.environ.get("MODEL_DIR") + 'glove.6B.50d.txt',
                action=Action.INSERT)
        ]

        cls.substitute_augmenters = [
            naw.Word2vecAug(
                model_path=os.environ.get("MODEL_DIR") + 'GoogleNews-vectors-negative300.bin',
                action=Action.SUBSTITUTE),
            naw.FasttextAug(
                model_path=os.environ.get("MODEL_DIR") + 'wiki-news-300d-1M.vec',
                action=Action.SUBSTITUTE),
            naw.GloVeAug(
                model_path=os.environ.get("MODEL_DIR") + 'glove.6B.50d.txt',
                action=Action.SUBSTITUTE)
        ]

    def test_oov(self):
        unknown_token = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'

        for aug in self.substitute_augmenters:
            augmented_text = aug.augment(unknown_token)
            self.assertEqual(unknown_token, augmented_text)

            text = unknown_token + ' the'

            augmented_text = aug.augment(text)
            self.assertNotEqual(text, augmented_text)
            self.assertTrue(unknown_token in augmented_text)

    def test_insert(self):
        texts = [
            'The quick brown fox jumps over the lazy dog'
        ]

        for aug in self.insert_augmenters:
            for text in texts:
                self.assertLess(0, len(text))
                augmented_text = aug.augment(text)

                self.assertLess(len(text.split(' ')), len(augmented_text.split(' ')))
                self.assertNotEqual(text, augmented_text)

        self.assertLess(0, len(texts))

    def test_substitute(self):
        texts = [
            'The quick brown fox jumps over the lazy dog'
        ]

        for aug in self.substitute_augmenters:
            for text in texts:
                self.assertLess(0, len(text))
                augmented_text = aug.augment(text)

                self.assertNotEqual(text, augmented_text)

        self.assertLess(0, len(texts))

    def test_bogus_fasttext_loading(self):
        import nlpaug.model.word_embs.fasttext as ft
        test_file = os.path.join(os.path.dirname(__file__), 'bogus_fasttext.vec')
        expected_vector =  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        fasttext = ft.Fasttext()
        fasttext.read(test_file)

        for word in fasttext.w2v:
            self.assertSequenceEqual(list(fasttext.w2v[word]), expected_vector)
        
        self.assertSequenceEqual(["test1", "test2", "test_3", "test 4", "test -> 5"], fasttext.vocab)

        self.assertEqual(len(fasttext.vectors), 5)

