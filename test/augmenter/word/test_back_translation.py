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
                'from_model_name': 'transformer.wmt19.en-ru', 
                'from_model_checkpt': 'model1.pt',
                'to_model_name': 'transformer.wmt19.ru-en',
                'to_model_checkpt': 'model1.pt'
            }, {
                'from_model_name': 'transformer.wmt18.en-de',
                'from_model_checkpt': 'wmt18.model1.pt', 
                'to_model_name': 'transformer.wmt19.de-en',
                'to_model_checkpt': 'model1.pt'
            }
        ]

    def test_back_translation(self):
        # From English
        texts = [
            self.text, 
            "Seeing all of the negative reviews for this movie, I figured that it could be yet another comic masterpiece that wasn't quite meant to be."
        ]
        for model_name in self.eng_model_names:
            aug = naw.BackTranslationAug(
                from_model_name=model_name['from_model_name'], from_model_checkpt=model_name['from_model_checkpt'],
                to_model_name=model_name['to_model_name'], to_model_checkpt=model_name['to_model_checkpt'])
            augmented_text = aug.augment(self.text)
            aug.clear_cache()
            self.assertNotEqual(self.text, augmented_text)

            augmented_texts = aug.augment(texts)
            aug.clear_cache()
            for d, a in zip(texts, augmented_texts):
                self.assertNotEqual(d, a)        

        self.assertTrue(len(self.eng_model_names) > 1)

    def test_load_from_local_path(self):
        base_model_dir = os.environ.get("MODEL_DIR")
        from_model_dir = os.path.join(base_model_dir, 'word', 'fairseq', 'wmt19.en-de')
        to_model_dir = os.path.join(base_model_dir, 'word', 'fairseq', 'wmt19.de-en', '')

        aug = naw.BackTranslationAug(
            from_model_name=from_model_dir, from_model_checkpt='model1.pt',
            to_model_name=to_model_dir, to_model_checkpt='model1.pt', is_load_from_github=False)

        augmented_text = aug.augment(self.text)
        aug.clear_cache()
        self.assertNotEqual(self.text, augmented_text)

    def test_load_from_local_path_inexist(self):
        from_model_dir = '/abc/'
        to_model_dir = '/def/'
        with self.assertRaises(ValueError) as error:
            aug = naw.BackTranslationAug(
                from_model_name=from_model_dir, from_model_checkpt='model1.pt',
                to_model_name=to_model_dir, to_model_checkpt='model1.pt', is_load_from_github=False)
        self.assertTrue('Cannot load model from local path' in str(error.exception))

        base_model_dir = os.environ.get("MODEL_DIR")
        from_model_dir = os.path.join(base_model_dir, 'word', 'fairseq', 'wmt19.en-de')
        to_model_dir = '/def/'
        with self.assertRaises(ValueError) as error:
            aug = naw.BackTranslationAug(
                from_model_name=from_model_dir, from_model_checkpt='model1.pt',
                to_model_name=to_model_dir, to_model_checkpt='model1.pt', is_load_from_github=False)
        self.assertTrue('Cannot load model from local path' in str(error.exception))
