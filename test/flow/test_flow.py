import unittest

import nlpaug.augmenter.char as nac
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
                              pipeline_p=0.5),
                naf.Sequential([
                    nac.OcrAug(), nac.QwertyAug(aug_min=1),
                    nac.RandomCharAug(action=Action.SUBSTITUTE, aug_min=1, aug_p=0.6)
                ], name='Sub_Seq')
            ]),
            naf.Sometimes([
                naf.Sometimes([nac.RandomCharAug(action=Action.INSERT),
                               nac.RandomCharAug(action=Action.DELETE)]),
                naf.Sequential([nac.OcrAug(), nac.QwertyAug(aug_min=1),
                                nac.RandomCharAug(action=Action.SUBSTITUTE, aug_min=1, aug_p=0.6)])
            ], pipeline_p=0.5)
        ]

        for flow in flows:
            for text in texts:
                tokens = text.split(' ')
                results = flow.augment(tokens)[0]

                at_least_one_not_equal = False
                for t, r in zip(tokens, results):
                    if t != r:
                        at_least_one_not_equal = True
                        break

                self.assertTrue(at_least_one_not_equal)
                self.assertLess(0, len(tokens))

        self.assertLess(0, len(texts))
