import unittest
import os
from dotenv import load_dotenv

import nlpaug.augmenter.word as naw


class TestWord(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env_config_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', '.env'))
        load_dotenv(env_config_path)

    def test_empty_input_for_insert(self):
        text = ' '

        augs = [
            naw.ContextualWordEmbsAug(action="insert"),
            naw.TfIdfAug(model_path=os.environ.get("MODEL_DIR"), action="substitute")
        ]

        for aug in augs:
            augmented_text = aug.augment(text)
            # FIXME: standardize return
            is_equal = augmented_text == '' or augmented_text == ' '
            self.assertTrue(is_equal)

    def test_empty_input_substitute(self):
        text = ' '
        augs = [
            naw.SpellingAug(dict_path=os.environ.get("MODEL_DIR") + 'spelling_en.txt')
        ]

        for aug in augs:
            augmented_text = aug.augment(text)
            self.assertEqual('', augmented_text)

    def test_empty_input_for_swap(self):
        texts = [' ']
        aug = naw.RandomWordAug(action="swap")
        for text in texts:
            augmented_text = aug.augment(text)

            self.assertEqual('', augmented_text)

        self.assertEqual(1, len(texts))

        tokens = [None]
        aug = naw.RandomWordAug(action="swap")
        for t in tokens:
            augmented_text = aug.augment(t)
            self.assertEqual(None, augmented_text)

        self.assertEqual(len(tokens), 1)

    def test_empty_input_for_delete(self):
        text = ' '
        # None
        augs = [
            naw.RandomWordAug(action="delete"),
            naw.RandomWordAug(action="delete", stopwords=['a', 'an', 'the'])
        ]

        for aug in augs:
            augmented_text = aug.augment(text)
            # FIXME: standardize return
            is_equal = augmented_text == '' or augmented_text == ' '
            self.assertTrue(is_equal)

    def test_skip_punctuation(self):
        text = '. . . . ! ? # @'

        augs = [
            naw.ContextualWordEmbsAug(action='insert'),
            naw.AntonymAug(),
            naw.TfIdfAug(model_path=os.environ.get("MODEL_DIR"), action="substitute")
        ]

        for aug in augs:
            augmented_text = aug.augment(text)
            self.assertEqual(text, augmented_text)

    def test_non_strip_input(self):
        text = ' Good boy '

        augs = [
            naw.ContextualWordEmbsAug(action='insert'),
            naw.AntonymAug(),
            naw.TfIdfAug(model_path=os.environ.get("MODEL_DIR"), action="substitute")
        ]

        for aug in augs:
            augmented_text = aug.augment(text)
            self.assertNotEqual(text, augmented_text)

    def test_excessive_space(self):
        # https://github.com/makcedward/nlpaug/issues/48
        text = 'The  quick brown fox        jumps over the lazy dog . 1  2 '
        expected_result = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '.', '1', '2']

        augs = [
            naw.ContextualWordEmbsAug(action='insert'),
            naw.AntonymAug(),
            naw.TfIdfAug(model_path=os.environ.get("MODEL_DIR"), action="substitute")
        ]

        for aug in augs:
            tokenized_text = aug._tokenizer(text)
            self.assertEqual(tokenized_text, expected_result)

    def test_multi_thread(self):
        text = 'The quick brown fox jumps over the lazy dog.'
        n = 3
        augs = [
            naw.RandomWordAug(),
            naw.WordEmbsAug(model_type='word2vec',
                            model_path=os.environ["MODEL_DIR"] + 'GoogleNews-vectors-negative300.bin'),
            naw.ContextualWordEmbsAug(
                model_path='xlnet-base-cased', action="substitute",
                skip_unknown_word=True, temperature=0.7, device='cpu')
        ]

        for num_thread in [1, 3]:
            for aug in augs:
                augmented_data = aug.augment(text, n=n, num_thread=num_thread)
                self.assertEqual(len(augmented_data), n)
