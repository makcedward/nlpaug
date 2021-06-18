import unittest
import os
from dotenv import load_dotenv

import nlpaug.augmenter.word as naw
import nlpaug.model.lang_models as nml


class TestBackTranslationAug(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env_config_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', '.env'))
        load_dotenv(env_config_path)

        cls.text = 'The quick brown fox jumps over the lazy dog'

        cls.eng_model_names = [{
                'from_model_name': 'Helsinki-NLP/opus-mt-en-de', 
                'to_model_name': 'Helsinki-NLP/opus-mt-de-en',
            }
        ]

    def sample_test_case(self, device):
        # From English
        texts = [
            self.text, 
            "Seeing all of the negative reviews for this movie, I figured that it could be yet another comic masterpiece that wasn't quite meant to be."
        ]
        for model_name in self.eng_model_names:
            aug = naw.BackTranslationAug(from_model_name=model_name['from_model_name'], 
                to_model_name=model_name['to_model_name'], device=device)
            augmented_text = aug.augment(self.text)
            aug.clear_cache()
            self.assertNotEqual(self.text, augmented_text)

            augmented_texts = aug.augment(texts)
            aug.clear_cache()
            for d, a in zip(texts, augmented_texts):
                self.assertNotEqual(d, a)

            if device == 'cpu':
                self.assertTrue(device == aug.model.get_device())
            elif 'cuda' in device:
                self.assertTrue('cuda' in aug.model.get_device())

        self.assertTrue(len(self.eng_model_names) > 1)

    def test_back_translation(self):
        if torch.cuda.is_available():
            self.sample_test_case('cuda')
        self.sample_test_case('cpu')
