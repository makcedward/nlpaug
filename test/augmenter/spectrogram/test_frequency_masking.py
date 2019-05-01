import unittest
import os
import numpy as np

from nlpaug.util.file.load import LoadUtil
from nlpaug.augmenter.spectrogram import FrequencyMaskingAug


class TestFrequencyMasking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        #https://freewavesamples.com/yamaha-v50-rock-beat-120-bpm
        cls.sample_wav_file = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', 'data', 'Yamaha-V50-Rock-Beat-120bpm.wav'))

    def test_substitute(self):
        freq_mask_para = 80

        mel_spectrogram = LoadUtil.load_mel_spectrogram(self.sample_wav_file, n_mels=128)
        aug = FrequencyMaskingAug(mask_factor=freq_mask_para)

        augmented_mel_spectrogram = aug.substitute(mel_spectrogram)

        self.assertEqual(len(mel_spectrogram[aug.model.f0]), np.count_nonzero(mel_spectrogram[aug.model.f0]))
        self.assertEqual(0, np.count_nonzero(augmented_mel_spectrogram[aug.model.f0]))
