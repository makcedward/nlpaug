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
            nas.FrequencyMaskingAug(),
            nas.TimeMaskingAug()
        ]

        for num_thread in [1, 3]:
            for aug in augs:
                augmented_data = aug.augment(mel_spectrogram, n=n, num_thread=num_thread)
                self.assertEqual(len(augmented_data), n)

    def test_zone_parameter(self):
        aug = nas.LoudnessAug(zone=(0, 1))
        aug = nas.LoudnessAug(zone=(0.5, 0.7))
        aug = nas.LoudnessAug(zone=(0.6, 1))

        with self.assertRaises(ValueError) as context:
            aug = nas.LoudnessAug(zone=(-1, 1))
        self.assertTrue('Lower bound of zone is smaller than' in str(context.exception))

        with self.assertRaises(ValueError) as context:
            aug = nas.LoudnessAug(zone=(0, 1.2))
        self.assertTrue('Upper bound of zone is larger than' in str(context.exception))

    def test_coverage_parameter(self):
        aug = nas.LoudnessAug(coverage=0)
        aug = nas.LoudnessAug(coverage=0.5)
        aug = nas.LoudnessAug(coverage=1)

        with self.assertRaises(ValueError) as context:
            aug = nas.LoudnessAug(coverage=-1)
        self.assertTrue('Coverage value should be between than 0 and 1 while' in str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            aug = nas.LoudnessAug(coverage=1.1)
        self.assertTrue('Coverage value should be between than 0 and 1 while' in str(context.exception))
