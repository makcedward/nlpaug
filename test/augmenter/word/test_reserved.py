import unittest

import nlpaug.augmenter.word as naw


class TestReserved(unittest.TestCase):
    def test_reserved_word(self):
        texts = [
            'Fwd: Mail for solution',
            'Dear NLP, Thanks. Regards. NLPAug'
        ]

        reserved_tokens = [
            ['Fwd', 'Forward'],
            ['Regards', 'Sincerely']
        ]
        aug = naw.ReservedAug(reserved_tokens=reserved_tokens)

        for text in texts:
            augmented_text = aug.augment(text)
            self.assertNotEqual(augmented_text, text)

    def test_only_match_word(self):
        text = 'Dear NLP, text, texttt Thanks. Regards NLPAug'
        reserved_tokens = [
            ['t', 'a']
        ]

        aug = naw.ReservedAug(reserved_tokens=reserved_tokens)
        augmented_text = aug.augment(text)
        self.assertEqual(augmented_text, text)

    def test_multi_words(self):
        texts = [
            'Dear NLP, Thanks. Best Regards Augmenter'
        ]

        reserved_tokens = [
            ['Best Regards', 'Sincerely', 'Regard'],
            ['NLP', 'Natural Langauge Processing', 'Text']
        ]
        aug = naw.ReservedAug(reserved_tokens=reserved_tokens)

        for text in texts:
            augmented_text = aug.augment(text)
            self.assertNotEqual(augmented_text, text)
            for t in ['NLP', 'Best Regards']:
                self.assertTrue(t not in augmented_text)

    def test_exact_match(self):
        texts = [
            'Dear NLP, Thanks. Regards NLPAug'
        ]

        reserved_tokens = [
            ['Best Regards', 'Sincerely', 'Regard']
        ]
        aug = naw.ReservedAug(reserved_tokens=reserved_tokens)
        for text in texts:
            augmented_text = aug.augment(text)

            self.assertEqual(augmented_text, text)

        reserved_tokens = [
            ['Regards', 'Sincerely', 'Regard']
        ]
        aug = naw.ReservedAug(reserved_tokens=reserved_tokens)
        for text in texts:
            augmented_text = aug.augment(text)

            self.assertNotEqual(augmented_text, text)

    def test_duplicate_word(self):
        text = 'Dear NLP, text, texttt Thanks. best regards NLPAug'
        reserved_tokens = [
            ['Best Regards', 'ABCD'],
            ['regards', '1234']
        ]

        aug = naw.ReservedAug(reserved_tokens=reserved_tokens, case_sensitive=False)
        augmented_text = aug.augment(text)
        self.assertTrue('ABCD' in augmented_text)

    def test_case_sentsitive(self):
        texts = [
            'Fwd: Mail for solution',
        ]

        reserved_tokens = [
            ['FWD', 'Forward'],
        ]
        aug = naw.ReservedAug(reserved_tokens=reserved_tokens, 
            case_sensitive=True)

        for text in texts:
            augmented_text = aug.augment(text)
            self.assertEqual(augmented_text, text)

        aug = naw.ReservedAug(reserved_tokens=reserved_tokens, 
            case_sensitive=False)

        for text in texts:
            augmented_text = aug.augment(text)
            self.assertNotEqual(augmented_text, text)

        text = 'Dear NLP, text, texttt Thanks. Regards NLPAug'
        reserved_tokens = [
            ['1', 'Regards'],
        ]

        aug = naw.ReservedAug(reserved_tokens=reserved_tokens, 
            case_sensitive=False)
        for _ in range(10):
            augmented_text = aug.augment(text)
            self.assertNotEqual(augmented_text, text)

        text = 'Dear NLP, text, texttt Thanks. regards NLPAug'
        reserved_tokens = [
            ['Best Regards', 'Regards']
        ]
        aug = naw.ReservedAug(reserved_tokens=reserved_tokens, 
            case_sensitive=False)
        for _ in range(10):
            augmented_text = aug.augment(text)
            self.assertNotEqual(augmented_text, text)
            self.assertTrue('Best Regards' in augmented_text)

        text = 'Dear NLP, text, texttt Thanks. best regards NLPAug'
        reserved_tokens = [
            ['Best Regards', 'Regards']
        ]
        aug = naw.ReservedAug(reserved_tokens=reserved_tokens, 
            case_sensitive=False)
        for _ in range(10):
            augmented_text = aug.augment(text)
            self.assertNotEqual(augmented_text, text)
            self.assertTrue('Regards' in augmented_text)
