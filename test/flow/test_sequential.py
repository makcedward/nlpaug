import unittest
import os
import numpy as np
from dotenv import load_dotenv

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.spectrogram as nas
import nlpaug.augmenter.audio as naa
import nlpaug.flow as naf
from nlpaug.util import Action, AudioLoader


class TestSequential(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env_config_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '.env'))
        load_dotenv(env_config_path)
        # https://freewavesamples.com/yamaha-v50-rock-beat-120-bpm
        cls.sample_wav_file = os.path.join(
            os.environ.get("TEST_DIR"), 'res', 'audio', 'Yamaha-V50-Rock-Beat-120bpm.wav'
        )

    def test_dry_run(self):
        flow = naf.Sequential()
        results = flow.augment([])
        self.assertEqual(0, len(results))

    def test_single_action(self):
        texts = [
            'The quick brown fox jumps over the lazy dog',
            'Zology raku123456 fasdasd asd4123414 1234584 s@#'
        ]

        flow = naf.Sequential([nac.RandomCharAug(action=Action.INSERT, min_char=1)])

        for text in texts:
            augmented_text = flow.augment(text)

            self.assertNotEqual(text, augmented_text)
            self.assertLess(0, len(text))

        self.assertLess(0, len(texts))

    def test_multiple_actions(self):
        texts = [
            'The quick brown fox jumps over the lazy dog',
            'Zology raku123456 fasdasd asd4123414 1234584'
        ]

        flows = [
            naf.Sequential([nac.RandomCharAug(action=Action.INSERT),
                            naw.RandomWordAug()]),
            naf.Sequential([nac.OcrAug(), nac.KeyboardAug(aug_char_min=1),
                            nac.RandomCharAug(action=Action.SUBSTITUTE, aug_char_min=1, aug_char_p=0.6, aug_word_p=0.6)])
        ]

        for flow in flows:
            for text in texts:
                augmented_text = flow.augment(text)

                self.assertNotEqual(text, augmented_text)
                self.assertLess(0, len(text))

            self.assertLess(0, len(texts))

        self.assertLess(0, len(flows))

    def test_spectrogram(self):
        mel_spectrogram = AudioLoader.load_mel_spectrogram(self.sample_wav_file, n_mels=128)

        flow = naf.Sequential([
            nas.FrequencyMaskingAug(stateless=False),
            nas.TimeMaskingAug(stateless=False),
            nas.TimeMaskingAug(stateless=False)])

        augmented_mel_spectrograms = flow.augment(mel_spectrogram)

        for aug in flow:
            if aug.name == 'FrequencyMasking_Aug':
                aug_data = augmented_mel_spectrograms[0][aug.f0:aug.f0+aug.f, aug.time_start:aug.time_end]
                orig_data = mel_spectrogram[aug.f0:aug.f0+aug.f, aug.time_start:aug.time_end]

                self.assertEqual(orig_data.size, np.count_nonzero(orig_data))
                self.assertEqual(0, np.count_nonzero(aug_data))
            elif aug.name == 'TimeMasking_Aug':
                self.assertEqual(len(mel_spectrogram[:, aug.t0]),
                                 np.count_nonzero(mel_spectrogram[:, aug.t0]))
                self.assertEqual(0, np.count_nonzero(augmented_mel_spectrograms[0][:, aug.t0]))
            else:
                raise ValueError('Unexpected flow for {} augmenter'.format(aug.name))

        self.assertTrue(len(flow) > 0)

    def test_audio(self):
        audio, sampling_rate = AudioLoader.load_audio(self.sample_wav_file)

        flow = naf.Sequential([
            naa.NoiseAug(),
            naa.PitchAug(sampling_rate=sampling_rate, factor=(0.2, 1.5)),
            naa.ShiftAug(sampling_rate=sampling_rate, duration=2),
            naa.SpeedAug(factor=(1.5, 3))
        ])

        augmented_audio = flow.augment(audio)

        self.assertFalse(np.array_equal(audio, augmented_audio))
        self.assertTrue(len(audio), len(augmented_audio))
