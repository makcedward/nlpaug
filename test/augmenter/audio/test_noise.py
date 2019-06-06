import unittest
import os
import librosa
import numpy as np

import nlpaug.augmenter.audio as naa


class TestNoise(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # https://freewavesamples.com/yamaha-v50-rock-beat-120-bpm
        cls.sample_wav_file = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', 'data', 'Yamaha-V50-Rock-Beat-120bpm.wav'))

    def test_empty_input(self):
        audio = np.array([])
        aug = naa.NoiseAug()
        augmented_audio = aug.augment(audio)

        self.assertTrue(np.array_equal(audio, augmented_audio))

    def test_subsitute(self):
        audio, sampling_rate = librosa.load(self.sample_wav_file)

        aug = naa.NoiseAug()
        augmented_audio = aug.augment(audio)

        self.assertFalse(np.array_equal(audio, augmented_audio))
        self.assertTrue(len(audio), len(augmented_audio))
