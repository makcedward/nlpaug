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
            'gpt2',
            'distilgpt2'
        ]

        cls.text = 'The quick brown fox jumps over the lazy'
        cls.texts = [
            'The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.'
            "Seeing all of the negative reviews for this movie, I figured that it could be yet another comic masterpiece that wasn't quite meant to be."
        ]

    def test_contextual_word_embs(self):
        # self.execute_by_device('cuda')
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

        augmented_text = aug.augment(text)
        self.assertEqual(text, augmented_text)

    def insert(self, aug, data):
        augmented_text = aug.augment(data)

        if isinstance(data, list):
            for d, a in zip(data, augmented_text):
                self.assertLess(len(d.split(' ')), len(a.split(' ')))
                self.assertNotEqual(d, a)
                self.assertTrue(aug.model.SUBWORD_PREFIX not in a)
        else:
            self.assertLess(len(data.split(' ')), len(augmented_text.split(' ')))
            self.assertNotEqual(data, augmented_text)
            self.assertTrue(aug.model.SUBWORD_PREFIX not in augmented_text)

    # def top_k(self, aug):
    #     original_top_k = aug.model.top_k

    #     aug.model.top_k = 10000

    #     augmented_text = aug.augment(self.text)

    #     self.assertNotEqual(self.text, augmented_text)

    #     self.assertLess(len(self.text.split(' ')), len(augmented_text.split(' ')))
    #     self.assertNotEqual(self.text, augmented_text)
    #     self.assertTrue(aug.model.SUBWORD_PREFIX not in augmented_text)

    #     aug.model.top_k = original_top_k

    # def top_p(self, aug):
    #     original_top_p = aug.model.top_p

    #     aug.model.top_p = 0.05

    #     for _ in range(20): # Make sure it can generate different result
    #         augmented_text = aug.augment(self.text)

    #         if augmented_text != self.text:
    #             break

    #     self.assertNotEqual(self.text, augmented_text)

    #     self.assertLess(len(self.text.split(' ')), len(augmented_text.split(' ')))
    #     self.assertNotEqual(self.text, augmented_text)
    #     self.assertTrue(aug.model.SUBWORD_PREFIX not in augmented_text)

    #     aug.model.top_p = original_top_p

    # def top_k_top_p(self, aug):
    #     original_top_k = aug.model.top_k
    #     original_top_p = aug.model.top_p

    #     aug.model.top_k = 10000
    #     aug.model.top_p = 0.005

    #     augmented_text = aug.augment(self.text)

    #     self.assertLess(len(self.text.split(' ')), len(augmented_text.split(' ')))
    #     self.assertNotEqual(self.text, augmented_text)
    #     self.assertTrue(aug.model.SUBWORD_PREFIX not in augmented_text)

    #     aug.model.top_k = original_top_k
    #     aug.model.top_p = original_top_p

    # def no_top_k_top_p(self, aug):
    #     original_top_k = aug.model.top_k
    #     original_top_p = aug.model.top_p

    #     aug.model.top_k = None
    #     aug.model.top_p = None

    #     augmented_text = aug.augment(self.text)

    #     self.assertNotEqual(self.text, augmented_text)

    #     self.assertLess(len(self.text.split(' ')), len(augmented_text.split(' ')))
    #     self.assertNotEqual(self.text, augmented_text)
    #     self.assertTrue(aug.model.SUBWORD_PREFIX not in augmented_text)

    #     aug.model.top_k = original_top_k
    #     aug.model.top_p = original_top_p

    def test_incorrect_model_name(self):
        with self.assertRaises(ValueError) as error:
            nas.ContextualWordEmbsForSentenceAug(model_path='unknown')

        self.assertTrue('Model name value is unexpected.' in str(error.exception))

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
            original_top_p = original_aug.model.top_p

            new_aug = nas.ContextualWordEmbsForSentenceAug(
                model_path=model_path, temperature=original_temperature+1, top_k=original_top_k+1,
                top_p=original_top_p+1)
            new_temperature = new_aug.model.temperature
            new_top_k = new_aug.model.top_k
            new_top_p = new_aug.model.top_p

            self.assertEqual(original_temperature+1, new_temperature)
            self.assertEqual(original_top_k + 1, new_top_k)
            self.assertEqual(original_top_p + 1, new_top_p)

    def test_optimize(self):
        model_paths = ['gpt2', 'distilgpt2']
        # model_paths = ['xlnet-base-cased']

        for model_path in model_paths:
            aug = nas.ContextualWordEmbsForSentenceAug(model_path=model_path)

            enable_optimize = aug.model.get_default_optimize_config()
            enable_optimize['external_memory'] = 1024
            disable_optimize = aug.model.get_default_optimize_config()
            disable_optimize['external_memory'] = 0

            original_optimize = aug.model.optimize

            aug.model.optimize = enable_optimize
            augmented_data = aug.augment(self.text)
            self.assertNotEqual(self.text, augmented_data)

            aug.model.optimize = disable_optimize
            augmented_data = aug.augment(self.text)
            self.assertNotEqual(self.text, augmented_data)

            aug.model.optimize = original_optimize
