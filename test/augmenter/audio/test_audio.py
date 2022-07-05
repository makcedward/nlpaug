import unittest
import os
from dotenv import load_dotenv

import nlpaug.augmenter.audio as naa
from nlpaug.util import AudioLoader


class TestAudio(unittest.TestCase):
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

    def test_multi_thread(self):
        n = 3
        augs = [
            naa.CropAug(sampling_rate=self.sampling_rate),
            naa.PitchAug(sampling_rate=self.sampling_rate)
        ]

        for num_thread in [1, 3]:
            for aug in augs:
                augmented_data = aug.augment(self.audio, n=n, num_thread=num_thread)
                self.assertEqual(len(augmented_data), n)

    def test_coverage_and_zone(self):
        params = [
            ((0.3, 0.7), 1),
            ((0, 1), 1)
        ]

        for zone, coverage in params:
            augs = [
                naa.LoudnessAug(zone=zone, coverage=coverage, stateless=False),
                naa.MaskAug(zone=zone, coverage=coverage, stateless=False),
                naa.NoiseAug(zone=zone, coverage=coverage, stateless=False),
                naa.PitchAug(zone=zone, coverage=coverage, stateless=False, sampling_rate=self.sampling_rate),
                naa.SpeedAug(zone=zone, coverage=coverage, stateless=False),
                naa.VtlpAug(zone=zone, coverage=coverage, stateless=False, sampling_rate=self.sampling_rate),
                naa.NormalizeAug(zone=zone, coverage=coverage, stateless=False),
                naa.PolarityInverseAug(zone=zone, coverage=coverage, stateless=False)
            ]

            for aug in augs:
                aug_data = aug.augment(self.audio)
                aug_audio = aug_data[0]
                self.assertTrue(len(aug_audio[aug.start_pos:aug.end_pos]), int(len(self.audio) * (zone[1] - zone[0]) * coverage))
