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

        cls.model_names = [{
                'from_model_name': 'transformer.wmt19.en-ru', 
                'from_model_checkpt': 'model1.pt',
                'to_model_name': 'transformer.wmt19.ru-en',
                'to_model_checkpt': 'model1.pt'
            }, {
                'from_model_name': 'transformer.wmt19.en-de', 
                'from_model_checkpt': 'model1.pt', 
                'to_model_name': 'transformer.wmt19.de-en',
                'to_model_checkpt': 'model1.pt'
            }
        ]

    def test_back_translation(self):
        for model_name in self.model_names:
            aug = naw.BackTranslationAug(
                from_model_name=model_name['from_model_name'], from_model_checkpt=model_name['from_model_checkpt'],
                to_model_name=model_name['to_model_name'], to_model_checkpt=model_name['to_model_checkpt'])
            augmented_text = aug.augment(self.text)
            aug.clear_cache()
            self.assertNotEqual(self.text, augmented_text)

        self.assertTrue(len(self.model_names) > 1)