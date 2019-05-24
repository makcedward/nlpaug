import unittest
import os
import librosa
import numpy as np

import nlpaug.augmenter.audio as naa


class TestSpeed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # https://freewavesamples.com/yamaha-v50-rock-beat-120-bpm
        cls.sample_wav_file = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', 'data', 'Yamaha-V50-Rock-Beat-120bpm.wav'))

    def testSubsitute(self):
        audio, sampling_rate = librosa.load(self.sample_wav_file)
        
        aug = naa.SpeedAug(speed_factor=1.5)
        augmented_audio = aug.substitute(audio)

        self.assertFalse(np.array_equal(audio, augmented_audio))
        self.assertTrue(len(audio), len(augmented_audio))
