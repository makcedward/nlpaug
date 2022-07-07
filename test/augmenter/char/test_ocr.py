import unittest, os

from nlpaug.augmenter.char import OcrAug


class TestOcr(unittest.TestCase):
    def test_ocr_single_word(self):
        texts = ['Zoology', 'roku123456']
        aug = OcrAug()
        for text in texts:
            augmented_data = aug.augment(text)
            augmented_text = augmented_data[0]
            self.assertNotEqual(text, augmented_text)

        self.assertTrue(len(texts) > 0)

    def test_ocr_single_word_nonexist_char(self):
        texts = ['AAAAA', 'KKKKK']
        aug = OcrAug()
        for text in texts:
            augmented_data = aug.augment(text)
            augmented_text = augmented_data[0]
            self.assertEqual(text, augmented_text)

        self.assertTrue(len(texts) > 0)

    def test_ocr_multi_words(self):
        texts = ['The quick brown fox jumps over the lazy dog']
        aug = OcrAug()

        for text in texts:
            # Since non-exist mapping word may be drawn, try several times
            is_augmented = False
            for _ in range(10):
                augmented_data = aug.augment(text)
                augmented_text = augmented_data[0]
                is_equal = text == augmented_text
                if not is_equal:
                    is_augmented = True
                    break

            self.assertTrue(is_augmented)

        self.assertTrue(len(texts) > 0)

    def test_ocr_model_from_dict(self):
        mapping = {'0': ['2']}
        aug = OcrAug(dict_of_path=mapping)
        augmented_data = aug.augment('0000000')
        augmented_text = augmented_data[0]
        self.assertIn('2', augmented_text)

    def test_ocr_model_from_json(self):
        sample_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'res', 'common', 'sample.json'))
        aug = OcrAug(dict_of_path=sample_path)
        augmented_data = aug.augment('0000000')
        augmented_text = augmented_data[0]
        self.assertIn('3', augmented_text)

        with self.assertRaises(Exception) as error:
            sample_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'res', 'common', 'non_exist.json'))
            aug = OcrAug(dict_of_path=sample_path)
        self.assertIn('The dict_of_path does not exist', str(error.exception))
        