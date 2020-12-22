import unittest
import os
import numpy as np
from dotenv import load_dotenv

from nlpaug import Augmenter


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
