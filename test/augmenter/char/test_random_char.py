import unittest
from collections import Counter
from unittest.mock import patch

from nlpaug.augmenter.char.random import RandomCharAug


class TestRandomCharReplaceAug(unittest.TestCase):
    def test_insert_single_word(self):
        texts = ['Zoology', 'roku123456']
        aug = RandomCharAug(action='insert', min_char=1)
        for text in texts:
            augmented_data = aug.augment(text)
            augmented_text = augmented_data[0]
            self.assertNotEqual(text, augmented_text)
            self.assertLess(len(text), len(augmented_text))

        self.assertTrue(len(texts) > 0)

    def test_insert_multi_words(self):
        texts = ['The quick brown fox jumps over the lazy dog']
        aug = RandomCharAug(action='insert', min_char=1)
        for text in texts:
            augmented_cnt = 0
            augmented_data = aug.augment(text)
            augmented_text = augmented_data[0]

            tokens = aug.tokenizer(text)
            augmented_tokens = aug.tokenizer(augmented_text)

            for token, augmented_token in zip(tokens, augmented_tokens):
                if token != augmented_token:
                    augmented_cnt += 1

            self.assertLessEqual(augmented_cnt, len(tokens))
            self.assertGreater(augmented_cnt, 0)
            self.assertNotEqual(text, augmented_text)
            self.assertLess(len(text), len(augmented_text))

        self.assertTrue(len(texts) > 0)

    def test_substitute_single_word(self):
        texts = ['Zoology', 'roku123456']
        aug = RandomCharAug(action='substitute', min_char=1)
        for text in texts:
            augmented_data = aug.augment(text)
            augmented_text = augmented_data[0]
            self.assertNotEqual(text, augmented_text)

        self.assertTrue(len(texts) > 0)

    def test_substitute_multi_words(self):
        texts = ['The quick brown fox jumps over the lazy dog']
        aug = RandomCharAug(action='substitute', min_char=1)
        for text in texts:
            augmented_cnt = 0
            augmented_data = aug.augment(text)
            augmented_text = augmented_data[0]

            tokens = aug.tokenizer(text)
            augmented_tokens = aug.tokenizer(augmented_text)

            for token, augmented_token in zip(tokens, augmented_tokens):
                if token != augmented_token:
                    augmented_cnt += 1

            self.assertLessEqual(augmented_cnt, len(tokens))
            self.assertGreater(augmented_cnt, 0)
            self.assertNotEqual(text, augmented_text)

        self.assertTrue(len(texts) > 0)

    def test_swap(self):
        texts = [
            'The quick brown fox jumps over the lazy dog',
            'testing'
        ]

        aug = RandomCharAug(action="swap", min_char=1)
        for text in texts:
            with patch.object(aug, '_get_aug_idxes', return_value=[0]), \
                    patch.object(aug, '_get_swap_position', return_value=1):
                augmented_text = aug.augment(text)[0]

            self.assertNotEqual(text, augmented_text)
            self.assertEqual(len(text), len(augmented_text))
            self.assertEqual(Counter(text.lower()), Counter(augmented_text.lower()))

        self.assertTrue(len(texts) > 0)

    def test_delete(self):
        tokens = ['Zoology', 'roku123456']
        aug = RandomCharAug(action='delete', min_char=1)
        for t in tokens:
            augmented_data = aug.augment(t)
            augmented_token = augmented_data[0]
            self.assertNotEqual(t, augmented_token)
            self.assertLess(len(augmented_token), len(t))

        self.assertTrue(len(tokens) > 0)

    def test_min_char(self):
        tokens = ['Zoology', 'roku123456']

        for action in ['insert', 'swap', 'substitute', 'delete']:
            aug = RandomCharAug(action=action, min_char=20)
            for t in tokens:
                augmented_data = aug.augment(t)
                augmented_token = augmented_data[0]
                self.assertEqual(t, augmented_token)
                self.assertEqual(len(augmented_token), len(t))

        self.assertTrue(len(tokens) > 0)

    def test_swap_middle(self):
        text = 'quick brown jumps over lazy'
        aug = RandomCharAug(action="swap", swap_mode='middle', min_char=4)

        with patch.object(aug, '_get_aug_idxes', return_value=[0]), \
                patch.object(aug, '_get_swap_position', return_value=1) as swap_position:
            augmented_text = aug.augment(text)[0]

        self.assertNotEqual(text, augmented_text)
        self.assertEqual(len(augmented_text), len(text))
        swap_position.assert_called_once_with(0, 4, mode='middle')

    def test_swap_random(self):
        text = 'quick brown jumps over lazy'
        aug = RandomCharAug(action="swap", swap_mode='random', min_char=4)

        with patch.object(aug, '_get_aug_idxes', return_value=[0]), \
                patch.object(aug, '_get_swap_position', return_value=1) as swap_position:
            augmented_text = aug.augment(text)[0]

        self.assertNotEqual(text, augmented_text)
        self.assertEqual(len(augmented_text), len(text))
        swap_position.assert_called_once_with(0, 4, mode='random')

    def test_candidates(self):
        candidates = ['AAA', '11', '===', '中文']
        text = 'quick brown jumps over lazy'
        aug = RandomCharAug(min_char=4, candidates=candidates)
        augmented_data = aug.augment(text)
        augmented_text = augmented_data[0]
        self.assertNotEqual(text, augmented_text)

        match = False
        for c in candidates:
            if c in augmented_text:
                match = True
                break

        self.assertTrue(match)

    def test_swap_only_selects_candidate_characters(self):
        aug = RandomCharAug(action='swap', min_char=1, candidates=['x'])

        self.assertEqual(['abcd'], aug.augment('abcd'))

    def test_delete_only_selects_candidate_characters(self):
        aug = RandomCharAug(
            action='delete', min_char=1, aug_char_p=1, candidates=['bc']
        )

        self.assertEqual(['ad'], aug.augment('abcd'))

    def test_delete_respects_special_character_candidates(self):
        aug = RandomCharAug(
            action='delete', min_char=1, spec_char='&', include_upper_case=False,
            include_lower_case=False, include_numeric=False,
        )

        self.assertEqual(['a%'], aug.augment('a%'))
