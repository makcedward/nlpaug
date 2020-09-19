import unittest
import os
import numpy as np
from dotenv import load_dotenv

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.audio as naa
import nlpaug.augmenter.spectrogram as nas
import nlpaug.flow as naf
from nlpaug.util.audio.loader import AudioLoader
from nlpaug.util import Action


class TestFlow(unittest.TestCase):
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
        flow = naf.Sequential([naf.Sequential()])
        results = flow.augment([])
        self.assertEqual(0, len(results))
    
    def test_multiple_actions(self):
        texts = [
            'The quick brown fox jumps over the lazy dog',
            'Zology raku123456 fasdasd asd4123414 1234584'
        ]
    
        flows = [
            naf.Sequential([
                naf.Sometimes([nac.RandomCharAug(action="insert"),
                               nac.RandomCharAug(action="delete")],
                              pipeline_p=0.9),
                naf.Sequential([
                    nac.RandomCharAug(action="substitute", aug_char_min=1, aug_char_p=0.6, aug_word_p=0.6)
                ], name='Sub_Seq')
            ]),
            naf.Sometimes([
                naf.Sometimes([nac.RandomCharAug(action="insert"),
                               nac.RandomCharAug(action="delete")]),
                naf.Sequential([nac.OcrAug(), nac.KeyboardAug(aug_char_min=1),
                                nac.RandomCharAug(action="substitute", aug_char_min=1, aug_char_p=0.6, aug_word_p=0.6)])
            ], pipeline_p=0.9)
        ]
    
        # Since prob may be low and causing do not perform data augmentation. Retry 5 times
        for flow in flows:
            for text in texts:
                at_least_one_not_equal = False
                for _ in range(5):
                    augmented_text = flow.augment(text, n=1)
    
                    if text != augmented_text:
                        at_least_one_not_equal = True
                        break
    
                self.assertTrue(at_least_one_not_equal)
        self.assertLess(0, len(flows))
        self.assertLess(0, len(texts))
    
    def test_n_output_textual(self):
        texts = [
            'The quick brown fox jumps over the lazy dog',
            'Zology raku123456 fasdasd asd4123414 1234584',
            'AAAAAAAAAAA AAAAAAAAAAAAAA'
        ]
        flows = [
            naf.Sequential([
                nac.RandomCharAug(action="insert"),
                naw.RandomWordAug()
            ]),
            naf.Sometimes([
                nac.RandomCharAug(action="insert"),
                nac.RandomCharAug(action="delete")
            ], pipeline_p=0.9),
            naf.Sequential([
                naf.Sequential([
                    nac.RandomCharAug(action="insert"),
                    naw.RandomWordAug()
                ]),
                naf.Sometimes([
                    nac.RandomCharAug(action="insert"),
                    nac.RandomCharAug(action="delete")
                ], pipeline_p=0.9)
            ])
        ]
    
        for flow in flows:
            for text in texts:
                augmented_texts = flow.augment(text, n=3)
                self.assertGreater(len(augmented_texts), 1)
                for augmented_text in augmented_texts:
                    self.assertNotEqual(augmented_text, text)
    
        self.assertLess(0, len(flows))
        self.assertLess(0, len(texts))
    
    def test_n_output_audio(self):
        audio, sampling_rate = AudioLoader.load_audio(self.sample_wav_file)
    
        flows = [
            naf.Sequential([
                naa.CropAug(sampling_rate=sampling_rate),
                naa.LoudnessAug()
            ]),
            naf.Sometimes([
                naa.CropAug(sampling_rate=sampling_rate),
                naa.LoudnessAug()
            ], pipeline_p=0.9),
            naf.Sequential([
                naf.Sequential([
                    naa.CropAug(sampling_rate=sampling_rate),
                    naa.LoudnessAug()
                ]),
                naf.Sometimes([
                    naa.CropAug(sampling_rate=sampling_rate),
                    naa.LoudnessAug()
                ], pipeline_p=0.9)
            ])
        ]
    
        for flow in flows:
            augmented_audios = flow.augment(audio, n=3)
            self.assertGreater(len(augmented_audios), 1)
            for augmented_audio in augmented_audios:
                self.assertFalse(np.array_equal(audio, augmented_audio))
    
        self.assertLess(0, len(flows))
    
    def test_n_output_spectrogram(self):
        mel_spectrogram = AudioLoader.load_mel_spectrogram(self.sample_wav_file, n_mels=128)
    #
        flows = [
            naf.Sequential([
                nas.FrequencyMaskingAug(),
                nas.TimeMaskingAug()
            ]),
            naf.Sometimes([
                nas.FrequencyMaskingAug(),
                nas.TimeMaskingAug()
            ], pipeline_p=0.9),
            naf.Sequential([
                naf.Sequential([
                    nas.FrequencyMaskingAug(),
                    nas.TimeMaskingAug()
                ]),
                naf.Sometimes([
                    nas.FrequencyMaskingAug(),
                    nas.TimeMaskingAug()
                ], pipeline_p=0.9)
            ])
        ]
    
        for flow in flows:
            augmented_mel_spectrograms = flow.augment(mel_spectrogram, n=3)
            self.assertGreater(len(augmented_mel_spectrograms), 1)
            for augmented_mel_spectrogram in augmented_mel_spectrograms:
                self.assertFalse(np.array_equal(mel_spectrogram, augmented_mel_spectrogram))
    
        self.assertLess(0, len(flows))
    
    def test_n_output_without_augmentation(self):
        texts = [
            'AAAAAAAAAAA AAAAAAAAAAAAAA'
        ]
        flows = [
            naf.Sequential([
                nac.OcrAug(),
                nac.OcrAug()
            ]),
            naf.Sometimes([
                nac.RandomCharAug(),
                nac.RandomCharAug()
            ], pipeline_p=0.00001)
        ]
    
        for flow in flows:
            for text in texts:
                for _ in range(5):
                    augmented_texts = flow.augment(text, n=3)
                    all_not_equal = False
                    for augmented_text in augmented_texts:
                        if augmented_text != text:
                            all_not_equal = True
                            break
                    if all_not_equal:
                        break
    
                self.assertFalse(all_not_equal)
        self.assertLess(0, len(flows))
        self.assertLess(0, len(texts))
    
    def test_multi_thread(self):
        text = 'The quick brown fox jumps over the lazy dog'
        n = 3

        w2v_model_path = os.path.join(os.environ["MODEL_DIR"], 'word', 'word_embs', 'GoogleNews-vectors-negative300.bin')

        flows = [
            naf.Sequential([
                naf.Sequential([
                    nac.OcrAug(),
                    naw.WordEmbsAug(
                        model_type='word2vec',
                        model_path=w2v_model_path)
                ]),
                naf.Sequential([
                    nac.RandomCharAug(),
                ]),
                naw.ContextualWordEmbsAug(
                    model_path='xlnet-base-cased', action="substitute",
                    temperature=0.7, device='cpu')
            ]),
            naf.Sometimes([
                naf.Sequential([
                    nac.OcrAug(),
                    nac.RandomCharAug(),
                ]),
                naf.Sometimes([
                    naw.WordEmbsAug(model_type='word2vec',
                                    model_path=w2v_model_path)
                ], pipeline_p=0.999),
                naw.ContextualWordEmbsAug(
                    model_path='xlnet-base-cased', action="substitute",
                    temperature=0.7, device='cpu')
            ], pipeline_p=0.9999)
        ]
    
        for num_thread in [1, 3]:
            for flow in flows:
                augmented_data = flow.augment(text, n=n, num_thread=num_thread)
                self.assertEqual(len(augmented_data), n)

    # def test_augment_detail(self):
    #     text = 'The quick brown fox jumps over the lazy dog'

    #     flows = [
    #         naf.Sequential([
    #             naf.Sometimes([nac.RandomCharAug(action="insert"),
    #                            nac.RandomCharAug(action="delete")],
    #                           pipeline_p=0.5),
    #             naf.Sequential([
    #                 nac.RandomCharAug(action="substitute", aug_char_min=1, aug_char_p=0.6, aug_word_p=0.6)
    #             ], name='Sub_Seq')
    #         ], include_detail=True),
    #         naf.Sometimes([
    #             naf.Sometimes([nac.RandomCharAug(action="insert"),
    #                            nac.RandomCharAug(action="delete")]),
    #             naf.Sequential([nac.OcrAug(), nac.KeyboardAug(aug_char_min=1),
    #                             nac.RandomCharAug(action="substitute", aug_char_min=1, aug_char_p=0.6, aug_word_p=0.6)])
    #         ], pipeline_p=1, include_detail=True)
    #     ]

    #     for flow in flows:
    #         augmented_text, augment_details = flow.augment(text)

    #         self.assertNotEqual(text, augmented_text)
    #         self.assertGreater(len(augment_details), 0)
    #         for augment_detail in augment_details:
    #             self.assertGreater(augment_detail['orig_start_pos'], -1)
    #             self.assertGreater(augment_detail['new_start_pos'], -1)
    #             self.assertGreater(augment_detail['change_seq'], 0)
    #             self.assertIn(augment_detail['action'], Action.getall())
