import unittest
import os
from dotenv import load_dotenv

import nlpaug.augmenter.sentence as nas
from nlpaug.util import Action, Doc


class TestSentence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env_config_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', '.env'))
        load_dotenv(env_config_path)

        cls.model_paths = [
            'xlnet-base-cased',
            'gpt2',
            'distilgpt2'
        ]

        cls.text = 'The quick brown fox jumps over the lazy dog.'

    def test_augment_detail(self):
        for model_path in self.model_paths:
            aug = nas.ContextualWordEmbsForSentenceAug(model_path=model_path, include_detail=True)

            augmented_text, augment_details = aug.augment(self.text)

            self.assertNotEqual(self.text, augmented_text)
            self.assertGreater(len(augment_details), 0)
            for augment_detail in augment_details:
                self.assertTrue(augment_detail['orig_token'] in self.text)
                self.assertEqual(augment_detail['orig_start_pos'], -1)
                self.assertGreater(augment_detail['new_start_pos'], -1)
                self.assertGreater(augment_detail['change_seq'], 0)
                self.assertIn(augment_detail['action'], Action.getall())

            self.assertNotEqual(self.text, augmented_text)
