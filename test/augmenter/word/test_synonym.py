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

        ppdb_model_file_path = os.path.join(os.environ.get("MODEL_DIR"), 'word', 'ppdb', 'ppdb-2.0-s-all')

        cls.augs = [
            naw.SynonymAug(aug_src='wordnet'),
            naw.SynonymAug(aug_src='ppdb', model_path=ppdb_model_file_path)
        ]

    def test_substitute(self):
        text = 'The quick brown fox jumps over the lazy dog'

        retry_cnt = 10
        for aug in self.augs:
            self.assertLess(0, len(text))

            passed = False
            for _ in range(retry_cnt):
                augmented_data = aug.augment(text)
                augmented_text = augmented_data[0]
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
                augmented_data = aug.augment(text)
                augmented_text = augmented_data[0]
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
        augmented_data = aug.augment(text)
        augmented_text = augmented_data[0]
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
            augmented_data = aug.augment(text)
            augmented_text = augmented_data[0]
            self.assertNotEqual(text, augmented_text)

        self.assertLess(0, len(texts))

        texts = [
            "a",
        ]

        for aug in self.augs:
            for text in texts:
                self.assertLess(0, len(text))
                augmented_data = aug.augment(text)
                augmented_text = augmented_data[0]
                self.assertEqual(text, augmented_text)

        self.assertLess(0, len(texts))

    def test_skip_punctuation(self):
        text = '. . . . ! ? # @'

        for aug in self.augs:
            augmented_data = aug.augment(text)
            augmented_text = augmented_data[0]
            self.assertEqual(text, augmented_text)

    def test_multilingual(self):
        # import nltk
        # nltk.download('omw')
        # French
        text = 'chien'
        expected_texts = [
            'cliquer', 'clic', 'aboyeur', 'hot dog', 'franc', 'canis familiaris', 'achille', 'toutou',
            'cliquet', 'clébard', 'talon', 'chienchien', 'quignon', 'chien de chasse']
        aug = naw.SynonymAug(aug_src='wordnet', lang='fra')
        augmented_data = aug.augment(text)
        augmented_text = augmented_data[0]
        self.assertTrue(augmented_text in expected_texts)

        expected_texts = [
            'toutou', 'maître chien', 'clébard', 'dog', 'chienne', 'chiens', 'chiot', 'cynophiles', 'clebs'
        ]
        model_path = os.path.join(os.environ.get("MODEL_DIR"), 'word', 'ppdb', 'ppdb-1.0-s-lexical-french')
        aug = naw.SynonymAug(aug_src='ppdb', model_path=model_path)
        augmented_data = aug.augment(text)
        augmented_text = augmented_data[0]
        self.assertTrue(augmented_text in expected_texts)

        # Spanish
        text = 'Un rápido zorro marrón salta sobre el perro perezoso'
        aug = naw.SynonymAug(aug_src='wordnet', lang='spa')
        for _ in range(10):
            augmented_data = aug.augment(text)
            augmented_text = augmented_data[0]
            if augmented_text != text:
                break

        self.assertNotEqual(augmented_text, text)

    # https://github.com/makcedward/nlpaug/issues/99
    def test_reload(self):
        text = 'The quick brown fox jumps over the lazy dog'

        aug = naw.SynonymAug(aug_src='wordnet')
        augmented_data = aug.augment(text)
        augmented_text = augmented_data[0]
        self.assertNotEqual(text, augmented_text)

        model_path = os.path.join(os.environ.get("MODEL_DIR"), 'word', 'ppdb', 'ppdb-2.0-s-all')
        aug2 = naw.SynonymAug(aug_src='ppdb', model_path=model_path)
        augmented_data = aug2.augment(text)
        augmented_text = augmented_data[0]
        self.assertNotEqual(text, augmented_text)
