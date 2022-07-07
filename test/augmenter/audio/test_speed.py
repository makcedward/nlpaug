import unittest
import os
from dotenv import load_dotenv

import nlpaug.augmenter.audio as naa
from nlpaug.util import AudioLoader


class TestSpeed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env_config_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', '.env'))
        load_dotenv(env_config_path)
        # https://freewavesamples.com/yamaha-v50-rock-beat-120-bpm
        cls.sample_wav_file = os.path.join(
            os.environ.get("TEST_DIR"), 'res', 'audio', 'Yamaha-V50-Rock-Beat-120bpm.wav'
        )
        cls.audio, cls.sampling_rate = AudioLoader.load_audio(cls.sample_wav_file)

    def test_substitute(self):
        for _ in range(10):
            aug = naa.SpeedAug(stateless=False)
            augmented_data = aug.augment(self.audio)
            augmented_audio = augmented_data[0]

            if aug.aug_factor < 1:
                self.assertGreater(len(augmented_audio), len(self.audio))
            else:
                self.assertLess(len(augmented_audio), len(self.audio))
