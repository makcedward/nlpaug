import unittest
import os
import librosa
import numpy as np
from dotenv import load_dotenv

import nlpaug.augmenter.audio as naa


class TestCrop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env_config_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', '.env'))
        load_dotenv(env_config_path)
        # https://freewavesamples.com/yamaha-v50-rock-beat-120-bpm
        cls.sample_wav_file = os.environ.get("DATA_DIR") + 'Yamaha-V50-Rock-Beat-120bpm.wav'

    def test_empty_input(self):
        audio = np.array([])
        aug = naa.CropAug(sampling_rate=44100)
        augmented_audio = aug.augment(audio)

        self.assertTrue(np.array_equal(audio, augmented_audio))

    def test_subsitute(self):
        audio, sampling_rate = librosa.load(self.sample_wav_file)

        aug = naa.CropAug(sampling_rate=sampling_rate)
        augmented_audio = aug.augment(audio)

        self.assertNotEqual(len(audio), len(augmented_audio))
