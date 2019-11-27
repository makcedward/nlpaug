import unittest
import os
import numpy as np
from dotenv import load_dotenv

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.augmenter.audio as naa
from nlpaug.util.audio import AudioLoader


class TestWordNet(unittest.TestCase):
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

        cls.textual_augs = [
            nac.RandomCharAug(),
            naw.ContextualWordEmbsAug(),
            nas.ContextualWordEmbsForSentenceAug()
        ]

        cls.audio_augs = [
            naa.CropAug(sampling_rate=cls.sampling_rate),
            naa.SpeedAug(),
        ]

    def test_textual_augmenter_n_output(self):
        text = 'The quick brown fox jumps over the lazy dog'
        n = 3
        for aug in self.textual_augs:
            augmented_texts = aug.augment(text, n=n)
            self.assertGreater(len(augmented_texts), 1)
            for augmented_text in augmented_texts:
                self.assertNotEqual(augmented_text, text)

    def test_textual_augmenter_n_output_thread(self):
        text = 'The quick brown fox jumps over the lazy dog'
        n = 3
        for aug in self.textual_augs:
            augmented_texts = aug.augments([text]*2, n=n, num_thread=n)
            self.assertGreater(len(augmented_texts), 1)
            for augmented_text in augmented_texts:
                self.assertNotEqual(augmented_text, text)

    def test_multiprocess_gpu(self):
        text = 'The quick brown fox jumps over the lazy dog'
        n = 3
        aug = naw.ContextualWordEmbsAug(force_reload=True, device='cuda')

        augmented_texts = aug.augment(text, n=n, num_thread=n)
        self.assertGreater(len(augmented_texts), 1)
        for augmented_text in augmented_texts:
            self.assertNotEqual(augmented_text, text)

    def test_audio_augmenter_n_output(self):
        n = 3
        for aug in self.audio_augs:
            augmented_audios = aug.augment(self.audio, n=n)
            self.assertGreater(len(augmented_audios), 1)
            for augmented_audio in augmented_audios:
                self.assertFalse(np.array_equal(augmented_audio, self.audio))

    def test_audio_augmenter_n_output_thread(self):
        n = 3
        for aug in self.audio_augs:
            augmented_audios = aug.augments([self.audio]*2, n=n, num_thread=n)
            self.assertGreater(len(augmented_audios), 1)
            for augmented_audio in augmented_audios:
                self.assertFalse(np.array_equal(augmented_audio, self.audio))
