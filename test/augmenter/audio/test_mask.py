import unittest
import os
import numpy as np
from dotenv import load_dotenv

import nlpaug.augmenter.audio as naa
from nlpaug.util import AudioLoader


class TestMask(unittest.TestCase):
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

    def test_empty_input(self):
        audio = np.array([])
        aug = naa.MaskAug(sampling_rate=44100)
        augmented_audio = aug.augment(audio)

        self.assertTrue(np.array_equal(audio, augmented_audio))

    def test_with_noise(self):
        aug = naa.MaskAug(sampling_rate=self.sampling_rate, mask_with_noise=True)
        augmented_audio = aug.augment(self.audio)

        self.assertFalse(np.array_equal(self.audio, augmented_audio))
        self.assertEqual(len(self.audio), len(augmented_audio))

    def test_without_noise(self):
        aug = naa.MaskAug(sampling_rate=self.sampling_rate, mask_with_noise=False)
        augmented_audio = aug.augment(self.audio)

        self.assertFalse(np.array_equal(self.audio, augmented_audio))
        self.assertEqual(len(self.audio), len(augmented_audio))

    def test_coverage(self):
        zone = (0.3, 0.7)
        coverage = 0.1

        aug = naa.MaskAug(sampling_rate=self.sampling_rate, zone=zone, coverage=coverage, mask_with_noise=False)
        aug.model.stateless = False
        augmented_audio = aug.augment(self.audio)

        reconstruct_augmented_audio = np.concatenate(
            (self.audio[:aug.model.start_pos], aug.model.aug_data, self.audio[aug.model.end_pos:]), axis=0)

        self.assertTrue(np.array_equal(augmented_audio, reconstruct_augmented_audio))
        self.assertTrue(len(aug.model.aug_data), int(len(self.audio) * (zone[1] - zone[0]) * coverage))

    def test_zone(self):
        zone = (0, 1)
        coverage = 1.

        aug = naa.MaskAug(sampling_rate=self.sampling_rate, zone=zone, coverage=coverage, mask_with_noise=False)
        aug.model.stateless = False
        augmented_audio = aug.augment(self.audio)

        reconstruct_augmented_audio = np.concatenate(
            (self.audio[:aug.model.start_pos], aug.model.aug_data, self.audio[aug.model.end_pos:]), axis=0)

        self.assertTrue(np.array_equal(augmented_audio, reconstruct_augmented_audio))
        self.assertTrue(len(aug.model.aug_data), int(len(self.audio) * (zone[1] - zone[0]) * coverage))