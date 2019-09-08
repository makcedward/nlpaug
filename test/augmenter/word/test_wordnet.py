import unittest

import nlpaug.augmenter.word as naw


class TestWordNet(unittest.TestCase):
    def test_substitute(self):
        texts = [
            'The quick brown fox jumps over the lazy dog'
        ]
        aug = naw.WordNetAug()

        for text in texts:
            self.assertLess(0, len(text))
            augmented_text = aug.augment(text)

            self.assertNotEqual(text, augmented_text)

        self.assertLess(0, len(texts))

    def test_stopwords(self):
        text = 'The quick brown fox jumps over the lazy dog'
        aug = naw.WordNetAug(stopwords=['quick', 'brown', 'fox'])

        self.assertLess(0, len(text))
        augmented_text = aug.augment(text)

        self.assertNotEqual(text, augmented_text)

    def test_no_separator(self):
        """
            Pull#11: Remove seperator (underscore/ hyphen)
        :return:
        """

        texts = [
            "linguistic"
        ]
        aug = naw.WordNetAug()

        for text in texts:
            self.assertLess(0, len(text))
            augmented_text = aug.augment(text)
            for separator in ['-', '_']:
                self.assertNotIn(separator, augmented_text)
            self.assertNotEqual(text, augmented_text)

        self.assertLess(0, len(texts))

    def test_single_word(self):
        """
            Issue#10: contains one character words like: 'I a'
        :return:
        """

        texts = [
            "I a",
            'I'
        ]
        aug = naw.WordNetAug()

        for text in texts:
            self.assertLess(0, len(text))
            augmented_text = aug.augment(text)
            self.assertNotEqual(text, augmented_text)

        self.assertLess(0, len(texts))

        texts = [
            "a",
        ]
        aug = naw.WordNetAug()

        for text in texts:
            self.assertLess(0, len(text))
            augmented_text = aug.augment(text)
            self.assertEqual(text, augmented_text)

        self.assertLess(0, len(texts))

    def test_antonyms(self):
        texts = [
            'Good bad'
        ]
        aug = naw.WordNetAug(is_synonym=False)

        for text in texts:
            self.assertLess(0, len(text))
            augmented_text = aug.augment(text)

            self.assertNotEqual(text, augmented_text)

        self.assertLess(0, len(texts))