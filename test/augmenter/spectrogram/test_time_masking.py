import unittest
import os
from dotenv import load_dotenv
import numpy as np

from nlpaug.util import AudioLoader
from nlpaug.augmenter.spectrogram import TimeMaskingAug


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

    def test_substitute(self):
        time_mask_para = 80

        mel_spectrogram = AudioLoader.load_mel_spectrogram(self.sample_wav_file, n_mels=self.num_of_freq_channel)
        aug = TimeMaskingAug(mask_factor=time_mask_para)

        augmented_mel_spectrogram = aug.augment(mel_spectrogram)

        self.assertEqual(len(mel_spectrogram[:, aug.model.t0]), np.count_nonzero(mel_spectrogram[:, aug.model.t0]))
        self.assertEqual(0, np.count_nonzero(augmented_mel_spectrogram[:, aug.model.t0]))
