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

    def test_allow_original(self):
        texts = [
            'Fwd: Mail for solution',
            'Dear NLP, Thanks. Regards. NLPAug'
        ]

        reserved_tokens = [
            ['Fwd', 'Forward'],
            ['Regards', 'Sincerely']
        ]
        aug = naw.ReservedAug(reserved_tokens=reserved_tokens,
            allow_original=True)

        for text in texts:
            at_least_one_true = False
            for _ in range(10):
                augmented_text = aug.augment(text)

                if augmented_text == text:
                    at_least_one_true = True
                    break
            
            self.assertTrue(at_least_one_true)

    def test_multi_words(self):
        texts = [
            'Dear NLP, Thanks. Best Regards NLPAug'
        ]

        reserved_tokens = [
            ['Best Regards', 'Sincerely', 'Regard']
        ]
        aug = naw.ReservedAug(reserved_tokens=reserved_tokens)

        for text in texts:
            augmented_text = aug.augment(text)

            self.assertNotEqual(augmented_text, text)

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

