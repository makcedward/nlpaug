import unittest
import os
import numpy as np
import librosa

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.spectrogram as nas
import nlpaug.augmenter.audio as naa
import nlpaug.flow as naf
from nlpaug.util import Action
from nlpaug.util.file.load import LoadUtil


class TestSequential(unittest.TestCase):
    def test_dry_run(self):
        flow = naf.Sequential()
        results = flow.augment([])
        self.assertEqual(0, len(results))

    def test_single_action(self):
        texts = [
            'The quick brown fox jumps over the lazy dog',
            'Zology raku123456 fasdasd asd4123414 1234584 s@#'
        ]

        flow = naf.Sequential([nac.RandomCharAug(action=Action.INSERT)])

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
            naf.Sequential([nac.OcrAug(), nac.KeyboardAug(aug_min=1),
                            nac.RandomCharAug(action=Action.SUBSTITUTE, aug_min=1, aug_char_p=0.6, aug_word_p=0.6)])
        ]

        for flow in flows:
            for text in texts:
                augmented_text = flow.augment(text)

                self.assertNotEqual(text, augmented_text)
                self.assertLess(0, len(text))

            self.assertLess(0, len(texts))

        self.assertLess(0, len(flows))

    def test_spectrogram(self):
        # https://freewavesamples.com/yamaha-v50-rock-beat-120-bpm
        sample_wav_file = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', 'data', 'Yamaha-V50-Rock-Beat-120bpm.wav'))

        mel_spectrogram = LoadUtil.load_mel_spectrogram(sample_wav_file, n_mels=128)

        flow = naf.Sequential([
            nas.FrequencyMaskingAug(mask_factor=50),
            nas.TimeMaskingAug(mask_factor=20),
            nas.TimeMaskingAug(mask_factor=30)])

        augmented_mel_spectrogram = flow.augment(mel_spectrogram)

        for aug in flow:
            if aug.name == 'FrequencyMasking_Aug':
                self.assertEqual(len(mel_spectrogram[aug.model.f0]), np.count_nonzero(mel_spectrogram[aug.model.f0]))
                self.assertEqual(0, np.count_nonzero(augmented_mel_spectrogram[aug.model.f0]))
            elif aug.name == 'TimeMasking_Aug':
                self.assertEqual(len(mel_spectrogram[:, aug.model.t0]),
                                 np.count_nonzero(mel_spectrogram[:, aug.model.t0]))
                self.assertEqual(0, np.count_nonzero(augmented_mel_spectrogram[:, aug.model.t0]))
            else:
                # Unexpected flow
                self.assertFalse(True)

        self.assertTrue(len(flow) > 0)

    def test_audio(self):
        # https://freewavesamples.com/yamaha-v50-rock-beat-120-bpm
        sample_wav_file = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', 'data', 'Yamaha-V50-Rock-Beat-120bpm.wav'))

        audio, sampling_rate = librosa.load(sample_wav_file)

        flow = naf.Sequential([
            naa.NoiseAug(),
            naa.PitchAug(sampling_rate=sampling_rate, pitch_range=(0.2, 1.5)),
            naa.ShiftAug(sampling_rate=sampling_rate, shift_max=2),
            naa.SpeedAug(speed_range=(1.5, 3))
        ])

        augmented_audio = flow.augment(audio)

        self.assertFalse(np.array_equal(audio, augmented_audio))
        self.assertTrue(len(audio), len(augmented_audio))
