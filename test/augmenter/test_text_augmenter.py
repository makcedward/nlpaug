import unittest
import os
import torch
import numpy as np
from dotenv import load_dotenv

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas


class TestTextAugmenter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env_config_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '.env'))
        load_dotenv(env_config_path)

        cls.augs = [
            nac.RandomCharAug(),
            naw.ContextualWordEmbsAug(),
            nas.ContextualWordEmbsForSentenceAug()
        ]

    def test_augmenter_n_output(self):
        text = 'The quick brown fox jumps over the lazy dog'
        n = 3
        for aug in self.augs:
            augmented_texts = aug.augment(text, n=n)
            self.assertGreater(len(augmented_texts), 1)
            for augmented_text in augmented_texts:
                self.assertNotEqual(augmented_text, text)

        for aug in self.augs:
            augmented_texts = aug.augment([text]*2, n=1, num_thread=1)
            self.assertGreater(len(augmented_texts), 1)
            for augmented_text in augmented_texts:
                self.assertNotEqual(augmented_text, text)

    def test_augmenter_n_output_thread(self):
        text = 'The quick brown fox jumps over the lazy dog'
        n = 3
        for aug in self.augs:
            augmented_texts = aug.augment([text]*2, n=n, num_thread=n)
            self.assertGreater(len(augmented_texts), 1)
            for augmented_text in augmented_texts:
                self.assertNotEqual(augmented_text, text)

    def test_multiprocess_gpu(self):
        text = 'The quick brown fox jumps over the lazy dog'
        n = 3
        if torch.cuda.is_available():
            aug = naw.ContextualWordEmbsAug(force_reload=True, device='cuda')

            augmented_texts = aug.augment(text, n=n, num_thread=n)
            self.assertGreater(len(augmented_texts), 1)
            for augmented_text in augmented_texts:
                self.assertNotEqual(augmented_text, text)

        self.assertTrue(True)

    def test_get_aug_range_idxes(self):
        aug = naw.RandomWordAug()
        self.assertTrue(len(aug._get_aug_range_idxes([])) == 0)