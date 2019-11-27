import unittest
import os
import numpy as np
from dotenv import load_dotenv

from nlpaug.util import AudioLoader
from nlpaug.augmenter.spectrogram import FrequencyMaskingAug


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
        mel_spectrogram = np.array([])
        aug = FrequencyMaskingAug(mask_factor=80)
        augmented_mel_spectrogram = aug.augment(mel_spectrogram)

        self.assertTrue(np.array_equal(np.array([]), augmented_mel_spectrogram))

    def test_substitute(self):
        freq_mask_para = 80

        mel_spectrogram = AudioLoader.load_mel_spectrogram(self.sample_wav_file, n_mels=128)
        aug = FrequencyMaskingAug(mask_factor=freq_mask_para)

        augmented_mel_spectrogram = aug.augment(mel_spectrogram)

        self.assertEqual(len(mel_spectrogram[aug.model.f0]), np.count_nonzero(mel_spectrogram[aug.model.f0]))
        self.assertEqual(0, np.count_nonzero(augmented_mel_spectrogram[aug.model.f0]))
