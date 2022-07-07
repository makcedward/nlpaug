import unittest
import os
import torch
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
            'gpt2',
            'distilgpt2'
        ]

        cls.text = 'The quick brown fox jumps over the lazy'
        cls.texts = [
            'The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.',
            "Seeing all of the negative reviews for this movie, I figured that it could be yet another comic masterpiece that wasn't quite meant to be."
        ]

    def test_batch_size(self):
        # 1 per batch
        aug = nas.ContextualWordEmbsForSentenceAug(model_path='distilgpt2', batch_size=1)
        aug_data = aug.augment(self.texts)
        self.assertEqual(len(aug_data), len(self.texts))

        # batch size = input size
        aug = nas.ContextualWordEmbsForSentenceAug(model_path='distilgpt2', batch_size=len(self.texts))
        aug_data = aug.augment(self.texts)
        self.assertEqual(len(aug_data), len(self.texts))

        # batch size > input size
        aug = nas.ContextualWordEmbsForSentenceAug(model_path='distilgpt2', batch_size=len(self.texts)+1)
        aug_data = aug.augment(self.texts)
        self.assertEqual(len(aug_data), len(self.texts))

        # input size > batch size
        # aug = nas.ContextualWordEmbsForSentenceAug(model_path='distilgpt2', batch_size=2)
        # aug_data = aug.augment(self.texts * 2)
        # self.assertEqual(len(aug_data), len(self.texts)*2)

    def test_none_device(self):
        for model_path in self.model_paths:
            aug = nas.ContextualWordEmbsForSentenceAug(
                model_path=model_path, force_reload=True, device=None)
            self.assertTrue(aug.device == 'cpu')

    def test_reset_model(self):
        for model_path in self.model_paths:
            original_aug = nas.ContextualWordEmbsForSentenceAug(model_path=model_path, top_p=0.5)
            original_temperature = original_aug.model.temperature
            original_top_k = original_aug.model.top_k
            # original_top_p = original_aug.model.top_p

            new_aug = nas.ContextualWordEmbsForSentenceAug(
                model_path=model_path, temperature=original_temperature+1, top_k=original_top_k+1)
            new_temperature = new_aug.model.temperature
            new_top_k = new_aug.model.top_k
            # new_top_p = new_aug.model.top_p

            self.assertEqual(original_temperature+1, new_temperature)
            self.assertEqual(original_top_k + 1, new_top_k)
            # self.assertEqual(original_top_p + 1, new_top_p)

    def test_by_device(self):
        if torch.cuda.is_available():
            self.execute_by_device('cuda')
        self.execute_by_device('cpu')

    def execute_by_device(self, device):
        for model_path in self.model_paths:
            aug = nas.ContextualWordEmbsForSentenceAug(model_path=model_path, device=device)

            self.empty_input(aug)

            for data in [self.text, self.texts]:
                self.insert(aug, data)

        self.assertLess(0, len(self.model_paths))

    def empty_input(self, aug):
        text = ''

        augmented_data = aug.augment(text)
        self.assertTrue(len(augmented_data) == 0)

    def insert(self, aug, data):
        augmented_data = aug.augment(data)

        if isinstance(data, list):
            for d, a in zip(data, augmented_data):
                self.assertLess(len(d.split(' ')), len(a.split(' ')))
                self.assertNotEqual(d, a)
        else:
            augmented_text = augmented_data[0]
            self.assertLess(len(data.split(' ')), len(augmented_text.split(' ')))
            self.assertNotEqual(data, augmented_text)
