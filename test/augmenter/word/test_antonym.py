import unittest
import os
from dotenv import load_dotenv

import nlpaug.augmenter.word as naw


class TestAntonym(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env_config_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', '.env'))
        load_dotenv(env_config_path)

        cls.augs = [
            naw.AntonymAug()
        ]

    def test_substitute(self):
        texts = [
            'Good bad',
            'Heart patients may benefit more from exercise than healthy people.',
            'Beer first or wine, either way might not be fine.'
        ]

        for aug in self.augs:
            for text in texts:
                for _ in range(5):
                    augmented_text = aug.augment(text)
                    self.assertNotEqual(text, augmented_text)

    def test_skip_punctuation(self):
        text = '. . . . ! ? # @'

        for aug in self.augs:
            augmented_text = aug.augment(text)
            self.assertEqual(text, augmented_text)
