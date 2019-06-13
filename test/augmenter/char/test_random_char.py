import unittest

from nlpaug.augmenter.char.random import RandomCharAug
from nlpaug.util import Action


class TestRandomCharReplaceAug(unittest.TestCase):
    def test_insert_single_word(self):
        texts = ['Zoology', 'roku123456']
        aug = RandomCharAug(action=Action.INSERT)
        for text in texts:
            augmented_text = aug.augment(text)
            self.assertNotEqual(text, augmented_text)
            self.assertLess(len(text), len(augmented_text))

        self.assertTrue(len(texts) > 0)

    def test_insert_multi_words(self):
        texts = ['The quick brown fox jumps over the lazy dog']
        aug = RandomCharAug(action=Action.INSERT)
        for text in texts:
            augmented_cnt = 0
            augmented_text = aug.augment(text)

            tokens = aug.tokenizer(text)
            augmented_tokens = aug.tokenizer(augmented_text)

            for token, augmented_token in zip(tokens, augmented_tokens):
                if token != augmented_token:
                    augmented_cnt += 1

            self.assertLess(augmented_cnt, len(tokens))
            self.assertNotEqual(text, augmented_text)
            self.assertLess(len(text), len(augmented_text))

        self.assertTrue(len(texts) > 0)

    def test_substitute_single_word(self):
        texts = ['Zoology', 'roku123456']
        aug = RandomCharAug(action=Action.SUBSTITUTE)
        for text in texts:
            augmented_text = aug.augment(text)
            self.assertNotEqual(text, augmented_text)

        self.assertTrue(len(texts) > 0)

    def test_substitute_multi_words(self):
        texts = ['The quick brown fox jumps over the lazy dog']
        aug = RandomCharAug(action=Action.SUBSTITUTE)
        for text in texts:
            augmented_cnt = 0
            augmented_text = aug.augment(text)

            tokens = aug.tokenizer(text)
            augmented_tokens = aug.tokenizer(augmented_text)

            for token, augmented_token in zip(tokens, augmented_tokens):
                if token != augmented_token:
                    augmented_cnt += 1

            self.assertLess(augmented_cnt, len(tokens))
            self.assertNotEqual(text, augmented_text)

        self.assertTrue(len(texts) > 0)

    def test_swap(self):
        texts = ['The quick brown fox jumps over the lazy dog']
        aug = RandomCharAug(action=Action.SWAP)
        for text in texts:
            augmented_cnt = 0
            augmented_text = aug.augment(text)

            tokens = aug.tokenizer(text)
            augmented_tokens = aug.tokenizer(augmented_text)

            for token, augmented_token in zip(tokens, augmented_tokens):
                if token != augmented_token:
                    augmented_cnt += 1

            self.assertLess(augmented_cnt, len(tokens))
            self.assertNotEqual(text, augmented_text)

        self.assertTrue(len(texts) > 0)

    def test_delete(self):
        tokens = ['Zoology', 'roku123456']
        aug = RandomCharAug(action=Action.DELETE)
        for t in tokens:
            augmented_text = aug.augment(t)
            self.assertNotEqual(t, augmented_text)
            self.assertLess(len(augmented_text), len(t))

        self.assertTrue(len(tokens) > 0)

