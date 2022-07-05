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
            'Older people feel more youthful when they also feel in control.',
            'Good bad',
            'Heart patients may benefit more from exercise than healthy people.',
            'Beer first or wine, either way might not be fine.'
        ]

        for aug in self.augs:
            for text in texts:
                for _ in range(5):
                    augmented_data = aug.augment(text)
                    augmented_text = augmented_data[0]
                    self.assertNotEqual(text, augmented_text)

    def test_unable_to_substitute(self):
        texts = [
            'Insomnia, sleep apnea diagnoses up sharply in U.S. Army.'
        ]

        for aug in self.augs:
            for text in texts:
                augmented_data = aug.augment(text)
                augmented_text = augmented_data[0]
                self.assertEqual(text, augmented_text)

    def test_skip_punctuation(self):
        text = '. . . . ! ? # @'

        for aug in self.augs:
            augmented_data = aug.augment(text)
            augmented_text = augmented_data[0]
            self.assertEqual(text, augmented_text)
