import unittest
import os
import sys
import types
import warnings
import numpy as np
import pytest
from contextlib import contextmanager
from dotenv import load_dotenv
from unittest.mock import patch

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


class FakeTqdm:
    calls = []

    def __init__(self, iterable, total=None, desc=None, leave=False):
        FakeTqdm.calls.append({
            'total': total,
            'desc': desc,
            'n_items': len(iterable) if hasattr(iterable, '__len__') else None,
        })
        self._iterable = iterable

    def __iter__(self):
        return iter(self._iterable)


@contextmanager
def fake_tqdm_installed():
    module = types.ModuleType('tqdm')
    module.tqdm = FakeTqdm
    FakeTqdm.calls = []
    with patch.dict(sys.modules, {'tqdm': module}):
        yield FakeTqdm


@contextmanager
def tqdm_missing():
    with patch.dict(sys.modules, {'tqdm': None}):
        yield


class TestAugmentProgressBar(unittest.TestCase):
    def test_disabled_by_default_leaves_results_unchanged(self):
        aug = DummyAugmenter(action=Action.SUBSTITUTE)
        self.assertEqual(['value|substitute'] * 2, aug.augment(' value ', n=2))
        self.assertEqual(
            ['alpha|substitute', 'beta|substitute'],
            aug.augment([' alpha ', ' beta '], progress_bar=False))

    def test_single_input_n_loop_shows_sized_bar(self):
        aug = DummyAugmenter(action=Action.SUBSTITUTE)
        with fake_tqdm_installed() as tqdm_cls:
            results = aug.augment(' value ', n=3, progress_bar=True)

        self.assertEqual(['value|substitute'] * 3, results)
        self.assertEqual(1, len(tqdm_cls.calls))
        self.assertEqual(3, tqdm_cls.calls[0]['total'])
        self.assertEqual(3, tqdm_cls.calls[0]['n_items'])

    def test_list_input_shows_sized_bar(self):
        aug = DummyAugmenter(action=Action.DELETE)
        with fake_tqdm_installed() as tqdm_cls:
            results = aug.augment([' alpha ', ' beta ', ' gamma '], progress_bar=True)

        self.assertEqual(['alpha|delete', 'beta|delete', 'gamma|delete'], results)
        self.assertEqual(1, len(tqdm_cls.calls))
        self.assertEqual(3, tqdm_cls.calls[0]['total'])

    def test_flow_augment_shows_sized_bar(self):
        import nlpaug.flow as naf

        flow = naf.Sequential([DummyAugmenter(action=Action.INSERT)])
        with fake_tqdm_installed() as tqdm_cls:
            results = flow.augment(' value ', n=2, progress_bar=True)

        self.assertEqual(2, len(results))
        self.assertTrue(all(isinstance(r, str) for r in results))
        self.assertEqual(2, tqdm_cls.calls[0]['total'])

    def test_missing_tqdm_warns_once_and_continues(self):
        import nlpaug.util.progress as progress_mod

        progress_mod._missing_tqdm_warned = False
        aug = DummyAugmenter(action=Action.SUBSTITUTE)
        with tqdm_missing():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                results = aug.augment(' value ', n=2, progress_bar=True)
                # Second call must not warn again
                results += aug.augment(' value ', n=1, progress_bar=True)

        self.assertEqual(['value|substitute'] * 3, results)
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        self.assertEqual(1, len(user_warnings))
        self.assertIn('tqdm', str(user_warnings[0].message))
        progress_mod._missing_tqdm_warned = False

    def test_augment_rejects_dataframe_and_series_issue_286(self):
        pd = pytest.importorskip('pandas')
        aug = DummyAugmenter(action=Action.SUBSTITUTE)
        with self.assertRaises(TypeError) as ctx:
            aug.augment(pd.DataFrame({'text': ['hello']}))
        self.assertIn('286', str(ctx.exception))
        with self.assertRaises(TypeError):
            aug.augment(pd.Series(['hello']))
