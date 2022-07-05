import unittest
import os
import numpy as np
from dotenv import load_dotenv

import nlpaug.augmenter.audio as naa
from nlpaug.util import AudioLoader


class TestNormalization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env_config_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', '.env'))
        load_dotenv(env_config_path)
        # https://freewavesamples.com/yamaha-v50-rock-beat-120-bpm
        cls.sample_wav_file = os.path.join(
            os.environ.get("TEST_DIR"), 'res', 'audio', 'Yamaha-V50-Rock-Beat-120bpm.wav'
        )
        cls.audio, cls.sampling_rate = AudioLoader.load_audio(cls.sample_wav_file)

    def test_empty_input(self):
        audio = np.array([])
        aug = naa.NormalizeAug()
        augmented_data = aug.augment(audio)

        self.assertTrue(np.array_equal(audio, augmented_data))

    def test_non_exist_method(self):
        with self.assertRaises(ValueError) as error:
            aug = naa.NormalizeAug(method='test1234')
        self.assertTrue('does not support yet. You may pick one' in str(error.exception))

    def test_minmax(self):
        aug = naa.NormalizeAug(method='minmax')
        augmented_data = aug.augment(self.audio)
        augmented_audio = augmented_data[0]

        self.assertFalse(np.array_equal(self.audio, augmented_audio))
        self.assertEqual(len(self.audio), len(augmented_audio))

    def test_max(self):
        aug = naa.NormalizeAug(method='max')
        augmented_data = aug.augment(self.audio)
        augmented_audio = augmented_data[0]

        self.assertFalse(np.array_equal(self.audio, augmented_audio))
        self.assertEqual(len(self.audio), len(augmented_audio))

    def test_standard(self):
        aug = naa.NormalizeAug(method='standard')
        augmented_data = aug.augment(self.audio)
        augmented_audio = augmented_data[0]

        self.assertFalse(np.array_equal(self.audio, augmented_audio))
        self.assertEqual(len(self.audio), len(augmented_audio))

    def test_random_method(self):
        aug = naa.NormalizeAug(method='random', stateless=False)
        augmented_data = aug.augment(self.audio)
        augmented_audio = augmented_data[0]

        self.assertTrue(aug.run_method in aug.model.get_support_methods())

        self.assertFalse(np.array_equal(self.audio, augmented_audio))
        self.assertEqual(len(self.audio), len(augmented_audio))
