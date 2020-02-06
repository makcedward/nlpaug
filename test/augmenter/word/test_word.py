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
        augs = [
            naw.RandomWordAug(),
            naw.WordEmbsAug(model_type='word2vec',
                            model_path=os.environ["MODEL_DIR"] + 'GoogleNews-vectors-negative300.bin'),
            naw.ContextualWordEmbsAug(
                model_path='xlnet-base-cased', action="substitute", device='cpu')
        ]

        for num_thread in [1, 3]:
            for aug in augs:
                augmented_data = aug.augment(text, n=num_thread, num_thread=num_thread)
                if num_thread == 1:
                    # return string
                    self.assertTrue(isinstance(augmented_data, str))
                else:
                    self.assertEqual(len(augmented_data), num_thread)

    def test_stopwords(self):
        text = 'The quick brown fox jumps over the lazy dog.'
        stopwords = ['The', 'brown', 'fox', 'jumps', 'the', 'dog']

        augs = [
            naw.RandomWordAug(action="delete", stopwords=stopwords),
            naw.ContextualWordEmbsAug(stopwords=stopwords),
            naw.WordEmbsAug(model_type='word2vec',
                            model_path=os.environ["MODEL_DIR"] + 'GoogleNews-vectors-negative300.bin',
                            stopwords=stopwords)
        ]

        for aug in augs:
            for i in range(10):
                augmented_text = aug.augment(text)
                self.assertTrue(
                    'quick' not in augmented_text or 'over' not in augmented_text or 'lazy' not in augmented_text)

    # https://github.com/makcedward/nlpaug/issues/81
    def test_stopwords_regex(self):
        text = 'The quick brown fox jumps over the lazy dog.'
        stopwords_regex = "( [a-zA-Z]{1}ox | [a-z]{1}og|(brown)|[a-zA-z]{1}he)|[a-z]{2}mps "

        augs = [
            naw.RandomWordAug(action="delete", stopwords_regex=stopwords_regex),
            naw.ContextualWordEmbsAug(stopwords_regex=stopwords_regex),
            naw.WordEmbsAug(model_type='word2vec',
                            model_path=os.environ["MODEL_DIR"] + 'GoogleNews-vectors-negative300.bin',
                            stopwords_regex=stopwords_regex)
        ]

        for aug in augs:
            for i in range(10):
                augmented_text = aug.augment(text)
                self.assertTrue(
                    'quick' not in augmented_text or 'over' not in augmented_text or 'lazy' not in augmented_text)

    # https://github.com/makcedward/nlpaug/issues/82
    def test_case(self):
        # Swap
        aug = naw.RandomWordAug(action='swap')
        self.assertEqual('bB aA', aug.augment('aA bB'))
        self.assertEqual(['Love', 'I', 'McDonalds'], aug.change_case('I love McDonalds'.split(' '), 1, 0))
        self.assertEqual(['Love', 'I', 'McDonalds'], aug.change_case('I love McDonalds'.split(' '), 0, 1))
        self.assertEqual(['Loves', 'he', 'McDonalds'], aug.change_case('He loves McDonalds'.split(' '), 1, 0))
        self.assertEqual(['Loves', 'he', 'McDonalds'], aug.change_case('He loves McDonalds'.split(' '), 0, 1))
        self.assertEqual(['He', 'McDonalds', 'loves'], aug.change_case('He loves McDonalds'.split(' '), 2, 1))

        # Insert
        aug = naw.TfIdfAug(model_path=os.environ.get("MODEL_DIR"), action='insert')
        expected = False
        for i in range(10):
            augmented_text = aug.augment('Good')
            if 'good' in augmented_text and aug.get_word_case(augmented_text.split(' ')[0]) == 'capitalize':
                expected = True
                break
        self.assertTrue(expected)

        # Substitute
        aug = naw.RandomWordAug(action='substitute', target_words=['abc'])
        expected = False
        for i in range(10):
            augmented_text = aug.augment('I love')
            if augmented_text == 'Abc love':
                expected = True
                break
        self.assertTrue(expected)

        aug = naw.AntonymAug()
        self.assertEqual('Unhappy', aug.augment('Happy'))

        # Do not change if target word is non-lower
        aug = naw.SpellingAug(dict_path=os.environ.get("MODEL_DIR") + 'spelling_en.txt')
        self.assertEqual('RE', aug.augment('Re'))

        # Delete case
        aug = naw.RandomWordAug(action='delete')
        expected = False
        for i in range(10):
            augmented_text = aug.augment('I love')
            if augmented_text == 'Love':
                expected = True
                break
        self.assertTrue(expected)

