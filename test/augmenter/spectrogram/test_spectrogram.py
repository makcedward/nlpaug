import unittest
import os
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

    def test_multi_thread(self):
        mel_spectrogram = AudioLoader.load_mel_spectrogram(self.sample_wav_file, n_mels=128)
        n = 3
        augs = [
            nas.FrequencyMaskingAug(mask_factor=80),
            nas.TimeMaskingAug(mask_factor=80)
        ]

        for num_thread in [1, 3]:
            for aug in augs:
                augmented_data = aug.augment(mel_spectrogram, n=n, num_thread=num_thread)
                self.assertEqual(len(augmented_data), n)
