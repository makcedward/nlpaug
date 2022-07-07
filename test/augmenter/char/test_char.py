import unittest

import nlpaug.augmenter.char as nac
import nlpaug.util.text.tokenizer as text_tokenizer
from nlpaug.util import Action


class TestCharacter(unittest.TestCase):
    def test_empty(self):
        texts = ['', None]

        augs = [
            nac.OcrAug(),
            nac.KeyboardAug(),
        ]

        for text in texts:
            for aug in augs:
                augmented_data = aug.augment(text)
                self.assertEqual(len(augmented_data), 0)

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

    def test_no_aug(self):
        aug = nac.KeyboardAug(aug_word_min=0.0, aug_word_p=0.05)
        text = '| 4 ||  || ½ || 0 || ½ || - || 1 || 1 || 1 || 0 || 0 || 0 || 1 || 1 || 1 || 1 || 1 || 1 || 10 || 67.75'

        augmented_data = aug.augment(text)
        self.assertEqual(text.replace(' ', ''), augmented_data[0].replace(' ', ''))

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

    def test_multi_inputs(self):
        texts = [
            'The quick brown fox jumps over the lazy dog.',
            'The quick brown fox jumps over the lazy dog.',
            'nac KeyboardAug ( tokenizer = text_tokenizer . split_sentence )',
            'nac KeyboardAug ( tokenizer = text_tokenizer . split_sentence )'
        ]
        augs = [
            nac.KeyboardAug(tokenizer=text_tokenizer.split_sentence),
            nac.RandomCharAug(tokenizer=text_tokenizer.split_sentence),
        ]

        num_thread = 2
        for aug in augs:
            augmented_data = aug.augment(texts, num_thread=num_thread)
            self.assertEqual(len(augmented_data), len(texts))

        num_thread = 1
        for aug in augs:
            augmented_data = aug.augment(texts, num_thread=num_thread)
            self.assertEqual(len(augmented_data), len(texts))

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
                augmented_data = aug.augment(text)
                augmented_text = augmented_data[0]
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
                augmented_data = aug.augment(text)
                augmented_text = augmented_data[0]
                self.assertTrue(
                    'quick' not in augmented_text or 'over' not in augmented_text or 'lazy' not in augmented_text)

    def test_min_char(self):
        text = 'He eats apple'
        augs = [
            nac.RandomCharAug(min_char=5),
            nac.KeyboardAug(min_char=5),
            nac.OcrAug(min_char=5)
        ]

        for aug in augs:
            augmented = False
            for i in range(10):
                augmented_data = aug.augment(text)
                augmented_text = augmented_data[0]
                if 'apple' not in augmented_text:
                    augmented = True
                    break

            self.assertTrue(augmented)

    def test_special_char(self):
        text = '#'
        aug = nac.KeyboardAug(min_char=1)
        augmented_data = aug.augment(text)
        augmented_text = augmented_data[0]
        self.assertNotEqual(text, augmented_text)

        # No mapping, return original value
        text = '~'
        augs = [
            nac.KeyboardAug(min_char=1),
            nac.OcrAug(min_char=1)
        ]
        for aug in augs:
            augmented_data = aug.augment(text)
            augmented_text = augmented_data[0]
            self.assertEqual(text, augmented_text)

    def test_empty_input_for_insert(self):
        texts = ['', '           ']
        augs = [
            nac.RandomCharAug(action='insert')
        ]

        for aug in augs:
            for text in texts:
                augmented_data = aug.augment(text)
                self.assertTrue(len(augmented_data) == 0 or augmented_data[0].strip() == '')

            augmented_texts = aug.augment(texts)
            for augmented_text in augmented_texts:
                self.assertTrue(len(augmented_text) == 0 or augmented_text.strip() == '')

    def test_empty_input_for_substitute(self):
        texts = ['', '           ']
        augs = [
            nac.RandomCharAug(action='substitute'),
            nac.KeyboardAug(),
            nac.OcrAug()
        ]

        for aug in augs:
            for text in texts:
                augmented_data = aug.augment(text)
                self.assertTrue(len(augmented_data) == 0 or augmented_data[0].strip() == '')

            augmented_texts = aug.augment(texts)
            for augmented_text in augmented_texts:
                self.assertTrue(len(augmented_text) == 0 or augmented_text.strip() == '')
