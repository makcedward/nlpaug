import unittest
import os
import numpy as np
from dotenv import load_dotenv

from nlpaug import Augmenter
from nlpaug.util import Action, Method


class TestBaseAugmenter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env_config_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '.env'))
        load_dotenv(env_config_path)

        cls.aug = Augmenter(name='base', method='flow', action='insert',
            aug_min=1, aug_max=10, aug_p=0.5)

    def test_generate_aug_cnt(self):
        self.assertEqual(0, self.aug.generate_aug_cnt(0))
        self.assertEqual(1, self.aug.generate_aug_cnt(1))
        self.assertGreater(self.aug.generate_aug_cnt(10), 1)


class DummyAugmenter(Augmenter):
    def __init__(self, action=Action.INSERT):
        super().__init__(
            name='dummy',
            method=Method.WORD,
            action=action,
            aug_min=1,
            aug_max=10,
            aug_p=0.5,
        )

    @classmethod
    def clean(cls, data):
        if isinstance(data, list):
            return [item.strip() if isinstance(item, str) else item for item in data]
        if isinstance(data, str):
            return data.strip()
        return data

    @classmethod
    def is_duplicate(cls, dataset, data):
        return data in dataset

    def insert(self, data):
        return f'{data}|insert'

    def substitute(self, data):
        return f'{data}|substitute'

    def swap(self, data):
        return f'{data}|swap'

    def delete(self, data):
        return f'{data}|delete'

    def crop(self, data):
        return f'{data}|crop'

    def split(self, data):
        return f'{data}|split'


class TestBaseAugmenterBehavior(unittest.TestCase):
    def test_get_action_handler_dispatches_all_supported_actions(self):
        expectations = {
            Action.INSERT: 'value|insert',
            Action.SUBSTITUTE: 'value|substitute',
            Action.SWAP: 'value|swap',
            Action.DELETE: 'value|delete',
            Action.CROP: 'value|crop',
            Action.SPLIT: 'value|split',
        }

        for action, expected in expectations.items():
            aug = DummyAugmenter(action=action)
            self.assertEqual(expected, aug.augment(' value ')[0])

    def test_single_input_multithread_returns_n_results(self):
        aug = DummyAugmenter(action=Action.SUBSTITUTE)
        results = aug.augment(' value ', n=3, num_thread=3)

        self.assertEqual(3, len(results))
        self.assertEqual(['value|substitute'] * 3, results)

    def test_list_input_multithread_uses_cleaned_values(self):
        aug = DummyAugmenter(action=Action.DELETE)
        results = aug.augment([' alpha ', ' beta '], num_thread=2)

        self.assertEqual(['alpha|delete', 'beta|delete'], results)

    def test_model_batch_augmenter_flattens_list_results(self):
        BatchAugmenter = type(
            'ContextualWordEmbsAug',
            (DummyAugmenter,),
            {
                'insert': lambda self, data: [f'{item}|batched' for item in data],
            },
        )

        aug = BatchAugmenter(action=Action.INSERT)
        results = aug.augment([' alpha ', ' beta '])

        self.assertEqual(['alpha|batched', 'beta|batched'], results)

    def test_empty_numpy_input_returns_empty_array(self):
        aug = DummyAugmenter(action=Action.INSERT)
        results = aug.augment(np.array([]))

        self.assertIsInstance(results, np.ndarray)
        self.assertEqual(0, results.size)
