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
            seq = naf.Sometimes([nac.RandomCharAug(action=Action.INSERT)], pipeline_p=0.6)
            for text in texts:
                tokens = text.split(' ')
                results = seq.augment(tokens)

                if len(results) < 1:
                    continue
                results = results[0]

                for t, r in zip(tokens, results):
                    if t != r:
                        at_least_one_not_equal = True
                        break

                if at_least_one_not_equal:
                    break

            if at_least_one_not_equal:
                break

        self.assertTrue(at_least_one_not_equal)
        self.assertLess(0, len(texts))

    def test_multiple_actions(self):
        texts = [
            'The quick brown fox jumps over the lazy dog',
            'Zology raku123456 fasdasd asd4123414 1234584'
        ]

        seqs = [
            naf.Sometimes([nac.RandomCharAug(action=Action.INSERT),
                           nac.RandomCharAug(action=Action.INSERT), nac.RandomCharAug(action=Action.DELETE)],
                          pipeline_p=0.8),
            naf.Sometimes(
                [nac.OcrAug(), nac.QwertyAug(aug_min=1),
                 nac.RandomCharAug(action=Action.SUBSTITUTE, aug_min=1, aug_p=0.4),
                 nac.RandomCharAug(action=Action.INSERT), nac.RandomCharAug(action=Action.DELETE)],
                pipeline_p=0.6)
        ]

        # Since prob may be low and causing do not perform data augmentation. Retry 5 times
        for seq in seqs:
            at_least_one_not_equal = False
            for _ in range(0, 5):
                for text in texts:
                    tokens = text.split(' ')
                    results = seq.augment(tokens)

                    if len(results) < 1:
                        continue

                    for t, r in zip(tokens, results):
                        if t != r:
                            at_least_one_not_equal = True
                            break

                if at_least_one_not_equal:
                    break

        self.assertTrue(at_least_one_not_equal)
        self.assertLess(0, len(seqs))
        self.assertLess(0, len(texts))

