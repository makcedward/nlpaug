import unittest

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.flow as naf
from nlpaug.util import Action


class TestFlow(unittest.TestCase):
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
                naf.Sometimes([nac.RandomCharAug(action=Action.INSERT),
                               nac.RandomCharAug(action=Action.DELETE)],
                              pipeline_p=0.9),
                naf.Sequential([
                    # nac.OcrAug(), nac.QwertyAug(aug_min=1),
                    nac.RandomCharAug(action=Action.SUBSTITUTE, aug_min=1, aug_char_p=0.6, aug_word_p=0.6)
                ], name='Sub_Seq')
            ]),
            naf.Sometimes([
                naf.Sometimes([nac.RandomCharAug(action=Action.INSERT),
                               nac.RandomCharAug(action=Action.DELETE)]),
                naf.Sequential([nac.OcrAug(), nac.QwertyAug(aug_min=1),
                                nac.RandomCharAug(action=Action.SUBSTITUTE, aug_min=1, aug_char_p=0.6, aug_word_p=0.6)])
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

    def test_n_output(self):
        texts = [
            'The quick brown fox jumps over the lazy dog',
            'Zology raku123456 fasdasd asd4123414 1234584',
            'AAAAAAAAAAA AAAAAAAAAAAAAA'
        ]
        flows = [
            naf.Sequential([
                nac.RandomCharAug(action=Action.INSERT),
                naw.RandomWordAug()
            ]),
            naf.Sometimes([
                nac.RandomCharAug(action=Action.INSERT),
                nac.RandomCharAug(action=Action.DELETE)
            ], pipeline_p=0.9),
            naf.Sequential([
                naf.Sequential([
                    nac.RandomCharAug(action=Action.INSERT),
                    naw.RandomWordAug()
                ]),
                naf.Sometimes([
                    nac.RandomCharAug(action=Action.INSERT),
                    nac.RandomCharAug(action=Action.DELETE)
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
                at_least_one_equal = False
                for _ in range(5):
                    augmented_texts = flow.augment(text, n=3)
                    if len(augmented_texts) == 1 and augmented_texts[0] == text:
                        at_least_one_equal = True
                        break

                self.assertTrue(at_least_one_equal)
        self.assertLess(0, len(flows))
        self.assertLess(0, len(texts))
