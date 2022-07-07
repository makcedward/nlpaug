import unittest
import os
import numpy as np
from dotenv import load_dotenv

from nlpaug.util import AudioLoader
import nlpaug.augmenter.spectrogram as nas


class TestFrequencyMasking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env_config_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', '.env'))
        load_dotenv(env_config_path)
        # https://freewavesamples.com/yamaha-v50-rock-beat-120-bpm
        cls.sample_wav_file = os.path.join(
            os.environ.get("TEST_DIR"), 'res', 'audio', 'Yamaha-V50-Rock-Beat-120bpm.wav'
        )

    def test_empty_input(self):
        data = np.array([])
        aug = nas.FrequencyMaskingAug()
        aug_data = aug.augment(data)

        self.assertTrue(np.array_equal(np.array([]), aug_data))

    def test_no_change_source(self):
        data = AudioLoader.load_mel_spectrogram(self.sample_wav_file, n_mels=128)
        aug = nas.FrequencyMaskingAug()
        aug_data = aug.augment(data)
        aug_audio = aug_data[0]

        comparison = data == aug_audio
        self.assertFalse(comparison.all())

    def test_substitute(self):
        data = AudioLoader.load_mel_spectrogram(self.sample_wav_file, n_mels=128)
        aug = nas.FrequencyMaskingAug(stateless=False)

        aug_data = aug.augment(data)
        aug_audio = aug_data[0]

        self.assertEqual(len(data[aug.f0]), np.count_nonzero(data[aug.f0]))
        self.assertEqual(0, np.count_nonzero(aug_audio[aug.f0][aug.time_start:aug.time_end]))
        self.assertEqual(0, len(np.where(aug_audio[aug.f0][:aug.time_start] == 0)[0]))
        self.assertEqual(0, len(np.where(aug_audio[aug.f0][aug.time_end:] == 0)[0]))
