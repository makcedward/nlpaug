import unittest

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.flow as naf
from nlpaug.util import Action


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
            naf.Sequential([nac.OcrAug(), nac.QwertyAug(aug_min=1),
                            nac.RandomCharAug(action=Action.SUBSTITUTE, aug_min=1, aug_p=0.6)])
        ]

        for flow in flows:
            for text in texts:
                augmented_text = flow.augment(text)

                self.assertNotEqual(text, augmented_text)
                self.assertLess(0, len(text))

            self.assertLess(0, len(texts))

        self.assertLess(0, len(flows))
