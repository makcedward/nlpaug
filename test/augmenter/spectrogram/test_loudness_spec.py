import unittest
import os
import numpy as np
from dotenv import load_dotenv

from nlpaug.util import AudioLoader
import nlpaug.augmenter.spectrogram as nas


class TestLoudnessSpec(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env_config_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', '.env'))
        load_dotenv(env_config_path)
        # https://freewavesamples.com/yamaha-v50-rock-beat-120-bpm
        cls.sample_wav_file = os.path.join(
            os.environ.get("TEST_DIR"), 'res', 'audio', 'Yamaha-V50-Rock-Beat-120bpm.wav'
        )

    def test_no_change_source(self):
        data = AudioLoader.load_mel_spectrogram(self.sample_wav_file, n_mels=128)
        aug = nas.LoudnessAug(stateless=False)
        aug_data = aug.augment(data)
        aug_audio = aug_data[0]

        comparison = data == aug_audio
        self.assertFalse(comparison.all())

    def test_substitute(self):
        data = AudioLoader.load_mel_spectrogram(self.sample_wav_file, n_mels=128)
        aug = nas.LoudnessAug(stateless=False)

        aug_data = aug.augment(data)
        aug_audio = aug_data[0]
        
        comparison = data[:, aug.time_start:aug.time_end] == aug_audio[:, aug.time_start:aug.time_end]
        self.assertFalse(comparison.all())
        comparison = data[:, :aug.time_start] == aug_audio[:, :aug.time_start]
        self.assertTrue(comparison.all())
        comparison = data[:, aug.time_end:] == aug_audio[:, aug.time_end:]
        self.assertTrue(comparison.all())
