import unittest
import os
import librosa
import numpy as np
from dotenv import load_dotenv

import nlpaug.augmenter.audio as naa


class TestAudio(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env_config_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', '.env'))
        load_dotenv(env_config_path)
        # https://freewavesamples.com/yamaha-v50-rock-beat-120-bpm
        cls.sample_wav_file = os.environ.get("DATA_DIR") + 'Yamaha-V50-Rock-Beat-120bpm.wav'

    def test_multi_thread(self):
        audio, sampling_rate = librosa.load(self.sample_wav_file)
        n = 3
        augs = [
            naa.CropAug(sampling_rate=sampling_rate),
            naa.PitchAug(sampling_rate=sampling_rate)
        ]

        for num_thread in [1, 3]:
            for aug in augs:
                augmented_data = aug.augment(audio, n=n, num_thread=num_thread)
                self.assertEqual(len(augmented_data), n)
