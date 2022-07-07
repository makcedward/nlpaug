import unittest
import os
import numpy as np
from dotenv import load_dotenv

import nlpaug.augmenter.audio as naa
from nlpaug.util import AudioLoader


class TestNoise(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env_config_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', '.env'))
        load_dotenv(env_config_path)
        # https://freewavesamples.com/yamaha-v50-rock-beat-120-bpm
        cls.sample_wav_file = os.path.join(
            os.environ.get("TEST_DIR"), 'res', 'audio', 'Yamaha-V50-Rock-Beat-120bpm.wav'
        )
        # https://en.wikipedia.org/wiki/Colors_of_noise
        cls.noise_wav_file = os.path.join(
            os.environ.get("TEST_DIR"), 'res', 'audio', 'Pink_noise.ogg'
        )
        cls.audio, cls.sampling_rate = AudioLoader.load_audio(cls.sample_wav_file)
        cls.noise, cls.noise_sampling_rate = AudioLoader.load_audio(cls.noise_wav_file)

    def test_empty_input(self):
        audio = np.array([])
        aug = naa.NoiseAug()
        augmented_data = aug.augment(audio)

        self.assertTrue(np.array_equal(audio, augmented_data))

    def test_substitute(self):
        aug = naa.NoiseAug()
        augmented_data = aug.augment(self.audio)
        augmented_audio = augmented_data[0]

        self.assertFalse(np.array_equal(self.audio, augmented_audio))
        self.assertTrue(len(self.audio), len(augmented_audio))
        self.assertTrue(self.sampling_rate > 0)

    def test_color_noise(self):
        colors = naa.NoiseAug().model.COLOR_NOISES

        for color in colors:
            aug = naa.NoiseAug(color=color)
            augmented_data = aug.augment(self.audio)
            augmented_audio = augmented_data[0]

            self.assertFalse(np.array_equal(self.audio, augmented_audio))
            self.assertTrue(len(self.audio), len(augmented_audio))
            self.assertTrue(self.sampling_rate > 0)

    def test_background_noise(self):
        # noise > audio
        aug = naa.NoiseAug(noises=[self.noise])
        augmented_data = aug.augment(self.audio)
        augmented_audio = augmented_data[0]
        self.assertTrue(augmented_audio is not None)

        # audio > noise
        aug = naa.NoiseAug(noises=[self.audio])
        augmented_data = aug.augment(self.audio)
        augmented_audio = augmented_data[0]
        self.assertTrue(augmented_audio is not None)
