import unittest
import os
from dotenv import load_dotenv

import nlpaug.augmenter.sentence as nas
from nlpaug.util import Action, Doc


class TestSentence(unittest.TestCase):
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

        cls.text = 'The quick brown fox jumps over the lazy dog.'
