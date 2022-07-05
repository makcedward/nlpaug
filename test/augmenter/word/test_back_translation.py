import unittest
import os
import torch
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
        cls.texts = [
            'The quick brown fox jumps over the lazy dog',
            "Seeing all of the negative reviews for this movie, I figured that it could be yet another comic masterpiece that wasn't quite meant to be."
        ]

        cls.eng_model_names = [{
                'from_model_name': 'facebook/wmt19-en-de',
                'to_model_name': 'facebook/wmt19-de-en',
            }
        ]

    def sample_test_case(self, device):
        # From English
        for model_name in self.eng_model_names:
            aug = naw.BackTranslationAug(from_model_name=model_name['from_model_name'], 
                to_model_name=model_name['to_model_name'], device=device)
            augmented_data = aug.augment(self.text)
            augmented_text = augmented_data[0]
            aug.clear_cache()
            self.assertNotEqual(self.text, augmented_text)

            augmented_texts = aug.augment(self.texts)
            aug.clear_cache()
            for d, a in zip(self.texts, augmented_texts):
                self.assertNotEqual(d, a)

            if device == 'cpu':
                self.assertTrue(device == aug.model.get_device())
            elif 'cuda' in device:
                self.assertTrue('cuda' in aug.model.get_device())

    def test_back_translation(self):
        if torch.cuda.is_available():
            self.sample_test_case('cuda')
        self.sample_test_case('cpu')

    def test_batch_size(self):
        model_name = self.eng_model_names[0]
        
        # 1 per batch
        aug = naw.BackTranslationAug(from_model_name=model_name['from_model_name'], 
            to_model_name=model_name['to_model_name'], batch_size=1)
        aug_data = aug.augment(self.texts)
        self.assertEqual(len(aug_data), len(self.texts))

        # batch size = input size
        aug = naw.BackTranslationAug(from_model_name=model_name['from_model_name'], 
            to_model_name=model_name['to_model_name'], batch_size=len(self.texts))
        aug_data = aug.augment(self.texts)
        self.assertEqual(len(aug_data), len(self.texts))

        # batch size > input size
        aug = naw.BackTranslationAug(from_model_name=model_name['from_model_name'], 
            to_model_name=model_name['to_model_name'], batch_size=len(self.texts)+1)
        aug_data = aug.augment(self.texts)
        self.assertEqual(len(aug_data), len(self.texts))

        # input size > batch size
        aug = naw.BackTranslationAug(from_model_name=model_name['from_model_name'], 
            to_model_name=model_name['to_model_name'], batch_size=2)
        aug_data = aug.augment(self.texts * 2)
        self.assertEqual(len(aug_data), len(self.texts)*2)