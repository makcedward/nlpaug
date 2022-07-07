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
            augmented_data = aug.augment(text)
            augmented_text = augmented_data[0]
            self.assertNotEqual(augmented_text, text)

    def test_only_match_word(self):
        text = 'Dear NLP, text, texttt Thanks. Regards NLPAug'
        reserved_tokens = [
            ['t', 'a']
        ]

        aug = naw.ReservedAug(reserved_tokens=reserved_tokens)
        augmented_data = aug.augment(text)
        augmented_text = augmented_data[0]
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
            augmented_data = aug.augment(text)
            augmented_text = augmented_data[0]

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
            augmented_data = aug.augment(text)
            augmented_text = augmented_data[0]

            self.assertEqual(augmented_text, text)

        reserved_tokens = [
            ['Regards', 'Sincerely', 'Regard']
        ]
        aug = naw.ReservedAug(reserved_tokens=reserved_tokens)
        for text in texts:
            augmented_data = aug.augment(text)
            augmented_text = augmented_data[0]

            self.assertNotEqual(augmented_text, text)

    def test_duplicate_word(self):
        text = 'Dear NLP, text, texttt Thanks. best regards NLPAug'
        reserved_tokens = [
            ['Best Regards', 'ABCD'],
            ['regards', '1234']
        ]

        aug = naw.ReservedAug(reserved_tokens=reserved_tokens, case_sensitive=False)
        augmented_data = aug.augment(text)
        augmented_text = augmented_data[0]
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
            augmented_data = aug.augment(text)
            augmented_text = augmented_data[0]
            self.assertEqual(augmented_text, text)

        aug = naw.ReservedAug(reserved_tokens=reserved_tokens, 
            case_sensitive=False)

        for text in texts:
            augmented_data = aug.augment(text)
            augmented_text = augmented_data[0]
            self.assertNotEqual(augmented_text, text)

        text = 'Dear NLP, text, texttt Thanks. Regards NLPAug'
        reserved_tokens = [
            ['1', 'Regards'],
        ]

        aug = naw.ReservedAug(reserved_tokens=reserved_tokens, 
            case_sensitive=False)
        for _ in range(10):
            augmented_data = aug.augment(text)
            augmented_text = augmented_data[0]
            self.assertNotEqual(augmented_text, text)

        text = 'Dear NLP, text, texttt Thanks. regards NLPAug'
        reserved_tokens = [
            ['Best Regards', 'Regards']
        ]
        aug = naw.ReservedAug(reserved_tokens=reserved_tokens, 
            case_sensitive=False)
        for _ in range(10):
            augmented_data = aug.augment(text)
            augmented_text = augmented_data[0]
            self.assertNotEqual(augmented_text, text)
            self.assertTrue('Best Regards' in augmented_text)

        text = 'Dear NLP, text, texttt Thanks. best regards NLPAug'
        reserved_tokens = [
            ['Best Regards', 'Regards']
        ]
        aug = naw.ReservedAug(reserved_tokens=reserved_tokens, 
            case_sensitive=False)
        for _ in range(10):
            augmented_data = aug.augment(text)
            augmented_text = augmented_data[0]
            self.assertNotEqual(augmented_text, text)
            self.assertTrue('Regards' in augmented_text)

    def test_all_combination_error(self):
        texts = [
            'Dear NLP, text, texttt Thanks. best regards NLPAug',
        ]
        reserved_tokens = [
            ['Best Regards', 'Best Regards2222', 'Best Regards3333', 'Best Regards4444'],
            ['thx', 'Thanks', 'thank you'],
            ['Dear', 'Hi', 'Hello']
        ]

        aug = naw.ReservedAug(
            aug_p=0.5,
            generate_all_combinations=True,
            reserved_tokens=reserved_tokens, 
            case_sensitive=False)

        with self.assertRaises(AssertionError) as error:
            aug.augment(texts)
        self.assertTrue('Augmentation probability has to be 1 to genenerate all combinations. Set aug_p=1 in constructor.' in str(error.exception))

    def test_all_combination(self):
        texts = [
            'Dear NLP, text, texttt Thanks. best regards NLPAug',
            'Dear Natural Language Processing, text, texttt Thanks, regards NLPAug'
        ]
        reserved_tokens = [
            ['Best Regards', 'Best Regards2222', 'Best Regards3333', 'Best Regards4444'],
            ['thx', 'Thanks', 'thank you'],
            ['Dear', 'Hi', 'Hello']
        ]
        aug = naw.ReservedAug(
            aug_p=1,
            generate_all_combinations=True,
            reserved_tokens=reserved_tokens, 
            case_sensitive=False)

        augmented_texts = aug.augment(texts)
        assert len(augmented_texts[0]) == 35
        assert len(augmented_texts[1]) == 8


