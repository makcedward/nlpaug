import unittest
import os
from dotenv import load_dotenv

import nlpaug.augmenter.word as naw


class TestSynonym(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env_config_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', '.env'))
        load_dotenv(env_config_path)

        cls.augs = [
            naw.SynonymAug(aug_src='wordnet'),
            naw.SynonymAug(aug_src='ppdb', model_path=os.environ.get("MODEL_DIR") + 'ppdb/ppdb-2.0-s-all.txt')
        ]

    def test_substitute(self):
        text = 'The quick brown fox jumps over the lazy dog'

        retry_cnt = 10
        for aug in self.augs:
            self.assertLess(0, len(text))

            passed = False
            for _ in range(retry_cnt):
                augmented_text = aug.augment(text)
                same_text = text == augmented_text

                if not same_text:
                    passed = True
                    break

            self.assertTrue(passed)

    def test_stopwords(self):
        text = 'The quick brown fox jumps over the lazy dog'

        retry_cnt = 10
        for aug in self.augs:
            original_stopwords = aug.stopwords
            aug.stopwords = ['quick', 'brown', 'fox']

            passed = False
            for _ in range(retry_cnt):
                augmented_text = aug.augment(text)
                same_text = text == augmented_text
                if not same_text:
                    passed = True
                    break

            self.assertTrue(passed)

            aug.stopwords = original_stopwords

    def test_no_separator_for_wordnet(self):
        """
            Pull#11: Remove seperator (underscore/ hyphen)
        :return:
        """

        text = "linguistic"

        aug = self.augs[0]  # WordNet only
        augmented_text = aug.augment(text)
        for separator in ['-', '_']:
            self.assertNotIn(separator, augmented_text)
        self.assertNotEqual(text, augmented_text)

    def test_single_word(self):
        """
            Issue#10: contains one character words like: 'I a'
        :return:
        """

        texts = [
            "I a",
            'I'
        ]

        aug = self.augs[0]  # WordNet only
        for text in texts:
            self.assertLess(0, len(text))
            augmented_text = aug.augment(text)
            self.assertNotEqual(text, augmented_text)

        self.assertLess(0, len(texts))

        texts = [
            "a",
        ]

        for aug in self.augs:
            for text in texts:
                self.assertLess(0, len(text))
                augmented_text = aug.augment(text)
                self.assertEqual(text, augmented_text)

        self.assertLess(0, len(texts))

    def test_skip_punctuation(self):
        text = '. . . . ! ? # @'

        for aug in self.augs:
            augmented_text = aug.augment(text)
            self.assertEqual(text, augmented_text)

    def test_multilingual(self):
        # French
        text = 'chien'
        expected_texts = [
            'cliquer', 'clic', 'aboyeur', 'hot dog', 'franc', 'canis familiaris', 'achille', 'toutou',
            'cliquet', 'clébard', 'talon', 'chienchien', 'quignon', 'chien de chasse']
        aug = naw.SynonymAug(aug_src='wordnet', lang='fra')
        augmented_text = aug.augment(text)
        self.assertTrue(augmented_text in expected_texts)

        expected_texts = [
            'toutou', 'maître chien', 'clébard', 'dog', 'chienne', 'chiens', 'chiot', 'cynophiles', 'clebs'
        ]
        aug = naw.SynonymAug(aug_src='ppdb', model_path=os.environ.get("MODEL_DIR") + 'ppdb/ppdb-1.0-s-lexical-french')
        augmented_text = aug.augment(text)
        self.assertTrue(augmented_text in expected_texts)

        # Spanish
        text = 'Un rápido zorro marrón salta sobre el perro perezoso'
        aug = naw.SynonymAug(aug_src='wordnet', lang='spa')
        for _ in range(10):
            augmented_text = aug.augment(text)
            if augmented_text != text:
                break

        self.assertNotEqual(augmented_text, text)
