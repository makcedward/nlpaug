import unittest

import nlpaug.augmenter.char as nac
import nlpaug.util.text.tokenizer as text_tokenizer


class TestCharacter(unittest.TestCase):
    def test_empty(self):
        texts = ['', None]

        augs = [
            nac.OcrAug(),
            nac.QwertyAug(),
        ]

        for text in texts:
            for aug in augs:
                augmented_text = aug.augment(text)
                self.assertEqual(text, augmented_text)

    def test_tokenizer(self):
        augs = [
            nac.OcrAug(tokenizer=text_tokenizer.split_sentence),
            nac.QwertyAug(tokenizer=text_tokenizer.split_sentence),
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