import unittest

from nlpaug.augmenter.char import OcrAug


class TestOcr(unittest.TestCase):
    def test_ocr_single_word(self):
        texts = ['Zoology', 'roku123456']
        aug = OcrAug()
        for text in texts:
            augmented_text = aug.augment(text)
            self.assertNotEqual(text, augmented_text)

        self.assertTrue(len(texts) > 0)

    def test_ocr_single_word_nonexist_char(self):
        texts = ['AAAAA', 'KKKKK']
        aug = OcrAug()
        for text in texts:
            augmented_text = aug.augment(text)
            self.assertEqual(text, augmented_text)

        self.assertTrue(len(texts) > 0)

    def test_ocr_multi_words(self):
        texts = ['The quick brown fox jumps over the lazy dog']
        aug = OcrAug()

        for text in texts:
            # Since non-exist mapping word may be drawn, try several times
            is_augmented = False
            for _ in range(10):
                augmented_text = aug.augment(text)
                is_equal = text == augmented_text
                if not is_equal:
                    is_augmented = True
                    break

            self.assertTrue(is_augmented)

        self.assertTrue(len(texts) > 0)
