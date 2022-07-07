import unittest
import os
from dotenv import load_dotenv
import numpy as np

from nlpaug.util import AudioLoader
import nlpaug.augmenter.spectrogram as nas


class TestTimeMasking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env_config_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', '.env'))
        load_dotenv(env_config_path)
        # https://freewavesamples.com/yamaha-v50-rock-beat-120-bpm
        cls.sample_wav_file = os.path.join(
            os.environ.get("TEST_DIR"), 'res', 'audio', 'Yamaha-V50-Rock-Beat-120bpm.wav'
        )
        cls.num_of_freq_channel = 128

    def test_no_change_source(self):
        data = AudioLoader.load_mel_spectrogram(self.sample_wav_file, n_mels=128)
        aug = nas.TimeMaskingAug()
        aug_data = aug.augment(data)

        comparison = data == aug_data
        self.assertFalse(comparison.all())

    def test_substitute(self):
        data = AudioLoader.load_mel_spectrogram(self.sample_wav_file, n_mels=self.num_of_freq_channel)
        aug = nas.TimeMaskingAug(stateless=False)

        aug_data = aug.augment(data)
        aug_audio = aug_data[0]

        self.assertEqual(len(data[:, aug.t0]), np.count_nonzero(data[:, aug.t0]))
        self.assertEqual(0, np.count_nonzero(aug_audio[:, aug.t0]))
