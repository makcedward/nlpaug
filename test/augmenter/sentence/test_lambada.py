import unittest
import os
import torch
import pandas as pd
from dotenv import load_dotenv

import nlpaug.augmenter.sentence as nas


class TestLambadaAug(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env_config_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', '.env'))
        load_dotenv(env_config_path)

        cls.model_dir = './model/lambada'

    def test_contextual_word_embs(self):
        if torch.cuda.is_available():
            self.execute_by_device('cuda')
        self.execute_by_device('cpu')

    def execute_by_device(self, device):
        aug = nas.LambadaAug(model_dir=self.model_dir, device=device, threshold=None, batch_size=2)

        labels = ['label0', 'label1', 'label2']
        self.insert(aug, labels)
        self.incorrect_label(aug, labels)
        self.batch_size(aug, labels)

        if device == 'cpu':
            self.assertTrue(device == aug.model.get_device())
        elif 'cuda' in device:
            self.assertTrue('cuda' in aug.model.get_device())

    def insert(self, aug, data):
        n = 3
        # test single input
        aug_data = aug.augment(data[0], n=n)
        self.assertEqual(n, len(aug_data))
        self.assertTrue(isinstance(aug_data, pd.DataFrame))

        # test multiple inputs
        aug_data = aug.augment(data, n=n)
        self.assertEqual(len(data)*n, len(aug_data))
        self.assertTrue(isinstance(aug_data, pd.DataFrame))

    def incorrect_label(self, aug, data):
        n = 1
        # single incorrect label
        incorrect_label = 'incorrect_label'
        with self.assertRaises(Exception) as error:
            aug.augment(incorrect_label, n=n)
        self.assertTrue('does not exist. Possible' in str(error.exception))

        # multi incorrect labels
        incorrect_labels = ['incorrect_label1', 'incorrect_label2']
        with self.assertRaises(Exception) as error:
            aug.augment(incorrect_labels, n=n)
        self.assertTrue('does not exist. Possible' in str(error.exception))
        
        # mix correct label and incorrect label
        with self.assertRaises(Exception) as error:
            aug.augment(data + incorrect_labels, n=n)
        self.assertTrue('does not exist. Possible' in str(error.exception))

    def batch_size(self, aug, data):
        n = 2
        orig_batch_size = aug.model.batch_size

        aug.model.batch_size = 1
        aug_data = aug.augment(data, n=n)
        self.assertEqual(len(data)*n, len(aug_data))

        aug.model.batch_size = len(data)
        aug_data = aug.augment(data, n=n)
        self.assertEqual(len(data)*n, len(aug_data))

        aug.model.batch_size = len(data)+1
        aug_data = aug.augment(data, n=n)
        self.assertEqual(len(data)*n, len(aug_data))

        aug.model.batch_size = orig_batch_size
