import unittest

import nlpaug.augmenter.char as nac
import nlpaug.util.text.tokenizer as text_tokenizer


class TestCharacter(unittest.TestCase):
    def test_empty(self):
        texts = ['', None]

        augs = [
            nac.OcrAug(),
            nac.KeyboardAug(),
        ]

        for text in texts:
            for aug in augs:
                augmented_text = aug.augment(text)
                self.assertEqual(text, augmented_text)

    def test_tokenizer(self):
        augs = [
            nac.OcrAug(tokenizer=text_tokenizer.split_sentence),
            nac.KeyboardAug(tokenizer=text_tokenizer.split_sentence),
            nac.RandomCharAug(tokenizer=text_tokenizer.split_sentence),
        ]

        text = 'The quick brown fox, jumps over lazy dog.'
        expected_tokens = ['The', ' quick', ' brown', ' fox', ', ', 'jumps', ' over', ' lazy', ' dog', '.']
        for aug in augs:
            tokens = aug.tokenizer(text)
            self.assertEqual(tokens, expected_tokens)

        text = 'The quick !brown fox, jumps # over lazy dog .'
        expected_tokens = ['The', ' quick', ' !', 'brown', ' fox', ', ', 'jumps', ' # ', 'over', ' lazy', ' dog', ' .']
        for aug in augs:
            tokens = aug.tokenizer(text)
            self.assertEqual(tokens, expected_tokens)

    def test_multi_thread(self):
        text = 'The quick brown fox jumps over the lazy dog.'
        n = 3
        augs = [
            nac.KeyboardAug(tokenizer=text_tokenizer.split_sentence),
            nac.RandomCharAug(tokenizer=text_tokenizer.split_sentence),
        ]

        for num_thread in [1, 3]:
            for aug in augs:
                augmented_data = aug.augment(text, n=n, num_thread=num_thread)
                self.assertEqual(len(augmented_data), n)

    def test_stopwords(self):
        text = 'The quick brown fox jumps over the lazy dog.'
        stopwords = ['The', 'brown', 'fox', 'jumps', 'the', 'dog']

        augs = [
            nac.RandomCharAug(stopwords=stopwords),
            nac.KeyboardAug(stopwords=stopwords),
            nac.OcrAug(stopwords=stopwords)
        ]

        for aug in augs:
            for i in range(10):
                augmented_text = aug.augment(text)
                self.assertTrue(
                    'quick' not in augmented_text or 'over' not in augmented_text or 'lazy' not in augmented_text)

    def test_stopwords_regex(self):
        text = 'The quick brown fox jumps over the lazy dog.'
        stopwords_regex = "( [a-zA-Z]{1}ox | [a-z]{1}og|(brown)|[a-zA-z]{1}he)|[a-z]{2}mps "

        augs = [
            nac.RandomCharAug(action="delete", stopwords_regex=stopwords_regex),
            nac.KeyboardAug(stopwords_regex=stopwords_regex),
            nac.OcrAug(stopwords_regex=stopwords_regex)
        ]

        for aug in augs:
            for i in range(10):
                augmented_text = aug.augment(text)
                self.assertTrue(
                    'quick' not in augmented_text or 'over' not in augmented_text or 'lazy' not in augmented_text)
