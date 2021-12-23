import unittest

import nlpaug.augmenter.char as nac
import nlpaug.flow as naf
from nlpaug.util import Action


class TestSometimes(unittest.TestCase):
    def test_dry_run(self):
        seq = naf.Sometimes()
        results = seq.augment([])
        self.assertEqual(0, len(results))

    def test_single_action(self):
        texts = [
            'The quick brown fox jumps over the lazy dog',
            'Zology raku123456 fasdasd asd4123414 1234584 s@#'
        ]

        # Since prob may be low and causing do not perform data augmentation. Retry 5 times
        at_least_one_not_equal = False
        for _ in range(0, 5):
            flow = naf.Sometimes([nac.RandomCharAug(action=Action.INSERT)], aug_p=0.6)
            for text in texts:
                augmented_text = flow.augment(text)

                if text != augmented_text:
                    at_least_one_not_equal = True

                self.assertLess(0, len(text))

            if at_least_one_not_equal:
                break

        self.assertTrue(at_least_one_not_equal)
        self.assertLess(0, len(texts))

    def test_multiple_actions(self):
        texts = [
            'The quick brown fox jumps over the lazy dog',
            'Zology raku123456 fasdasd asd4123414 1234584'
        ]

        flows = [
            naf.Sometimes([nac.RandomCharAug(action=Action.INSERT),
                           nac.RandomCharAug(action=Action.INSERT), nac.RandomCharAug(action=Action.DELETE)],
                          aug_p=0.8),
            naf.Sometimes(
                [nac.OcrAug(), nac.KeyboardAug(aug_char_min=1),
                 nac.RandomCharAug(action=Action.SUBSTITUTE, aug_char_min=1, aug_char_p=0.6, aug_word_p=0.6),
                 nac.RandomCharAug(action=Action.INSERT), nac.RandomCharAug(action=Action.DELETE)],
                aug_p=0.6)
        ]

        # Since prob may be low and causing do not perform data augmentation. Retry 5 times
        for flow in flows:
            at_least_one_not_equal = False
            for _ in range(0, 5):
                for text in texts:
                    self.assertLess(0, len(text))
                    augmented_text = flow.augment(text)

                    if text != augmented_text:
                        at_least_one_not_equal = True

                    self.assertLess(0, len(text))

                if at_least_one_not_equal:
                    break

        self.assertTrue(at_least_one_not_equal)
        self.assertLess(0, len(flows))
        self.assertLess(0, len(texts))

