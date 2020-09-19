import unittest
import os
import numpy as np
from dotenv import load_dotenv

import nlpaug.augmenter.audio as naa
from nlpaug.util.audio import AudioLoader


class TestAudioAugmenter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env_config_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '.env'))
        load_dotenv(env_config_path)
        # https://freewavesamples.com/yamaha-v50-rock-beat-120-bpm
        cls.sample_wav_file = os.path.join(
            os.environ.get("TEST_DIR"), 'res', 'audio', 'Yamaha-V50-Rock-Beat-120bpm.wav'
        )
        cls.audio, cls.sampling_rate = AudioLoader.load_audio(cls.sample_wav_file)

        cls.audio_augs = [
            naa.CropAug(sampling_rate=cls.sampling_rate),
            naa.SpeedAug(),
        ]

    def test_augmenter_n_output(self):
        n = 3
        for aug in self.audio_augs:
            augmented_audios = aug.augment(self.audio, n=n)
            self.assertEqual(len(augmented_audios), n)
            for augmented_audio in augmented_audios:
                self.assertFalse(np.array_equal(augmented_audio, self.audio))

        data = [self.audio, self.audio, self.audio]
        for aug in self.audio_augs:
            augmented_audios = aug.augment(data, n=1)
            self.assertEqual(len(augmented_audios), len(data))
            for d, augmented_audio in zip(data, augmented_audios):
                self.assertFalse(np.array_equal(augmented_audio, d))

    def test_augmenter_n_output_thread(self):
        n = 3
        for aug in self.audio_augs:
            augmented_audios = aug.augment([self.audio]*2, n=n, num_thread=n)
            self.assertGreater(len(augmented_audios), 1)
            for augmented_audio in augmented_audios:
                self.assertFalse(np.array_equal(augmented_audio, self.audio))
