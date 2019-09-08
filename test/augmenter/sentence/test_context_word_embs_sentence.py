import unittest
import os
from dotenv import load_dotenv

import nlpaug.augmenter.sentence as nas


class TestContextualWordEmbsAug(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env_config_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', '.env'))
        load_dotenv(env_config_path)

        cls.model_paths = [
            'xlnet-base-cased',
            'gpt2'
        ]

    def test_contextual_word_embs(self):
        self.execute_by_device('cuda')
        self.execute_by_device('cpu')

    def execute_by_device(self, device):
        for model_path in self.model_paths:
            aug = nas.ContextualWordEmbsForSentenceAug(model_path=model_path, force_reload=True, device=device)

            self.empty_input(aug)
            self.insert(aug)
            self.top_k_top_p(aug)

        self.assertLess(0, len(self.model_paths))

    def empty_input(self, aug):
        text = ''

        augmented_text = aug.augment(text)
        self.assertEqual(text, augmented_text)

    def insert(self, aug):
        text = 'The quick brown fox jumps over the lazy dog.'

        augmented_text = aug.augment(text)

        self.assertLess(len(text.split(' ')), len(augmented_text.split(' ')))
        self.assertNotEqual(text, augmented_text)
        self.assertTrue(aug.model.SUBWORD_PREFIX not in augmented_text)

    def top_k_top_p(self, aug):
        text = 'The quick brown fox jumps over the lazy dog.'
        original_top_k = aug.model.top_k
        original_top_p = aug.model.top_p

        aug.model.top_k = 10000
        aug.model.top_p = 0.005

        augmented_text = aug.augment(text)

        self.assertLess(len(text.split(' ')), len(augmented_text.split(' ')))
        self.assertNotEqual(text, augmented_text)
        self.assertTrue(aug.model.SUBWORD_PREFIX not in augmented_text)

        aug.model.top_k = original_top_k
        aug.model.top_p = original_top_p

    def test_incorrect_model_name(self):
        with self.assertRaises(ValueError) as error:
            nas.ContextualWordEmbsForSentenceAug(model_path='unknown')

        self.assertTrue('Model name value is unexpected.' in str(error.exception))
