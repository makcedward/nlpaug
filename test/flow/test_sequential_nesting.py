import random
import unittest

import numpy as np

import nlpaug.augmenter.char as nac
import nlpaug.flow as naf


class TestSequentialSometimesNesting(unittest.TestCase):
    """Regression test for issue #340.

    Chaining Sometimes flows inside Sequential used to produce nested
    outputs like [['text']] instead of ['text'], and on older releases even
    raised `AttributeError: 'list' object has no attribute 'strip'`.
    """

    def test_sequential_of_sometimes_stays_flat(self):
        data = 'abc def ghi jkl mno pqr stu wxy z 01234567890 [] '
        bad_shapes = []
        for seed in range(100):
            random.seed(seed)
            np.random.seed(seed)
            s1 = naf.Sometimes(
                nac.RandomCharAug(action='insert', aug_char_min=1, aug_char_p=0.3), aug_p=0.5)
            s2 = naf.Sometimes(
                nac.RandomCharAug(action='delete', aug_char_min=1, aug_char_p=0.3), aug_p=0.5)
            flow = naf.Sequential([s1, s2])
            out = flow.augment(data)
            if not (isinstance(out, list) and len(out) == 1 and isinstance(out[0], str)):
                bad_shapes.append((seed, out))
        self.assertEqual([], bad_shapes)

    def test_three_sometimes_chain_does_not_crash(self):
        # Mirrors the structure reported in #340 (Sometimes x3 in Sequential)
        for _ in range(20):
            s1 = naf.Sometimes(
                nac.RandomCharAug(action='insert', aug_char_min=1, aug_char_p=0.05), aug_p=0.5)
            s2 = naf.Sometimes(
                nac.RandomCharAug(action='insert', aug_char_min=1, aug_char_p=0.05), aug_p=0.5)
            s3 = naf.Sometimes(
                nac.RandomCharAug(action='delete', aug_char_min=1, aug_char_p=0.05), aug_p=0.5)
            flow = naf.Sequential([s1, s2, s3])
            out = flow.augment('abc def ghi jkl mno pqr stu wxy z 01234567890 [] ')
            self.assertIsInstance(out, list)
            self.assertEqual(1, len(out))
            self.assertIsInstance(out[0], str)

    def test_augment_result_can_be_fed_back(self):
        # The nested output previously broke downstream augments that
        # expected a string (the original AttributeError in #340).
        flow = naf.Sequential([
            naf.Sometimes(nac.RandomCharAug(action='insert', aug_char_min=1, aug_char_p=0.3), aug_p=0.5),
            naf.Sometimes(nac.RandomCharAug(action='delete', aug_char_min=1, aug_char_p=0.3), aug_p=0.5),
        ])
        out = flow.augment('abc def ghi')
        self.assertIsInstance(out[0], str)
        # Feeding the single result back through another flow must not crash
        out2 = flow.augment(out[0])
        self.assertIsInstance(out2[0], str)


if __name__ == '__main__':
    unittest.main()
