import unittest
import os
import re
from dotenv import load_dotenv
import sklearn.datasets

import nlpaug.augmenter.word as naw
import nlpaug.model.word_stats as nmw
from nlpaug.util import Action


class TestTfIdf(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env_config_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', '.env'))
        load_dotenv(env_config_path)

    def test_all(self):
        self._train()

        self._empty_input_for_insert()
        self._oov()
        self._insert()
        self._substitute()
        self._substitute_stopwords()
        self._skip_punctuation()

        self.housekeep()

    def housekeep(self):
        if os.path.exists('tfidfaug_w2idf.txt'):
            os.remove('tfidfaug_w2idf.txt')
        if os.path.exists('tfidfaug_w2tfidf.txt'):
            os.remove('tfidfaug_w2tfidf.txt')

    def _train(self):
        def _tokenizer(text, token_pattern=r"(?u)\b\w\w+\b"):
            token_pattern = re.compile(token_pattern)
            return token_pattern.findall(text)

        train_data = sklearn.datasets.fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
        train_x = train_data.data

        train_x_tokens = [_tokenizer(x) for x in train_x]

        tfidf_model = nmw.TfIdf()
        tfidf_model.train(train_x_tokens)
        tfidf_model.save('.')

        aug = naw.TfIdfAug(model_path='.', tokenizer=_tokenizer)

        texts = [
            'The quick brown fox jumps over the lazy dog',
            'asdasd test apple dog asd asd'
        ]

        for text in texts:
            augmented_text = aug.augment(text)
            self.assertNotEqual(text, augmented_text)

        self.assertLess(0, len(texts))

    def _empty_input_for_insert(self):
        texts = [' ']
        aug = naw.TfIdfAug(model_path=os.environ.get("MODEL_DIR"), action=Action.SUBSTITUTE)

        for text in texts:
            augmented_data = aug.augment(text)
            augmented_text = augmented_data[0]

            self.assertEqual('', augmented_text)

        self.assertEqual(1, len(texts))

    def _oov(self):
        unknown_token = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
        texts = [
            unknown_token,
            unknown_token + ' the'
        ]

        augmenters = [
            naw.TfIdfAug(model_path=os.environ.get("MODEL_DIR"), action=Action.INSERT),
            naw.TfIdfAug(model_path=os.environ.get("MODEL_DIR"), action=Action.SUBSTITUTE)
        ]

        for aug in augmenters:
            for text in texts:
                self.assertLess(0, len(text))
                augmented_data = aug.augment(text)
                augmented_text = augmented_data[0]
                if aug.action == Action.INSERT:
                    self.assertLess(len(text.split(' ')), len(augmented_text.split(' ')))
                    self.assertNotEqual(text, augmented_text)
                elif aug.action == Action.SUBSTITUTE:
                    self.assertEqual(len(text.split(' ')), len(augmented_text.split(' ')))

                    if unknown_token == text:
                        self.assertEqual(text, augmented_text)
                    else:
                        self.assertNotEqual(text, augmented_text)
                else:
                    raise Exception('Augmenter is neither INSERT or SUBSTITUTE')

    def _insert(self):
        texts = [
            'The quick brown fox jumps over the lazy dog'
        ]

        aug = naw.TfIdfAug(model_path=os.environ.get("MODEL_DIR"), action=Action.INSERT)

        for i in range(20):

            for text in texts:
                self.assertLess(0, len(text))
                augmented_data = aug.augment(text)
                augmented_text = augmented_data[0]

                self.assertLess(len(text.split(' ')), len(augmented_text.split(' ')))
                self.assertNotEqual(text, augmented_text)

            self.assertLess(0, len(texts))

    def _substitute(self):
        texts = [
            'The quick brown fox jumps over the lazy dog'
        ]

        aug = naw.TfIdfAug(model_path=os.environ.get("MODEL_DIR"), action=Action.SUBSTITUTE)

        for text in texts:
            self.assertLess(0, len(text))
            augmented_data = aug.augment(text)
            augmented_text = augmented_data[0]

            self.assertNotEqual(text, augmented_text)

        self.assertLess(0, len(texts))

    def _substitute_stopwords(self):
        texts = [
            'The quick brown fox jumps over the lazy dog'
        ]

        stopwords = [t.lower() for t in texts[0].split(' ')[:3]]
        aug_n = 3

        aug = naw.TfIdfAug(
            model_path=os.environ.get("MODEL_DIR"), action=Action.SUBSTITUTE,
            stopwords=stopwords)

        for text in texts:
            self.assertLess(0, len(text))
            augmented_data = aug.augment(text)
            augmented_text = augmented_data[0]

            augmented_tokens = aug.tokenizer(augmented_text)
            tokens = aug.tokenizer(text)

            augmented_cnt = 0

            for token, augmented_token in zip(tokens, augmented_tokens):
                if token.lower() in stopwords and len(token) > aug_n:
                    self.assertEqual(token.lower(), augmented_token)
                else:
                    augmented_cnt += 1

            self.assertGreater(augmented_cnt, 0)

        self.assertLess(0, len(texts))

    def _skip_punctuation(self):
        text = '. . . . ! ? # @'

        aug = naw.TfIdfAug(
            model_path=os.environ.get("MODEL_DIR"),
            action=Action.SUBSTITUTE)

        augmented_data = aug.augment(text)
        augmented_text = augmented_data[0]
        self.assertEqual(text, augmented_text)
