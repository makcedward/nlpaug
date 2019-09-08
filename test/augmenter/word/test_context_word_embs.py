import unittest
import os
from dotenv import load_dotenv

import nlpaug.augmenter.word as naw
import nlpaug.model.lang_models as nml
from nlpaug.util import Action


class TestContextualWordEmbsAug(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env_config_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', '.env'))
        load_dotenv(env_config_path)

        cls.model_paths = [
            'bert-base-uncased',
            'xlnet-base-cased'
        ]

    def test_contextual_word_embs(self):
        self.execute_by_device('cuda')
        self.execute_by_device('cpu')

    def execute_by_device(self, device):
        for model_path in self.model_paths:
            insert_aug = naw.ContextualWordEmbsAug(
                model_path=model_path, action="insert", force_reload=True, device=device)
            substitute_aug = naw.ContextualWordEmbsAug(
                model_path=model_path, action="substitute", force_reload=True, device=device)

            self.oov([insert_aug, substitute_aug])
            self.insert(insert_aug)
            self.substitute(substitute_aug)
            self.substitute_stopwords(substitute_aug)
            self.subword([insert_aug, substitute_aug])
            self.not_substitute_unknown_word(substitute_aug)
            self.top_k_top_p([insert_aug, substitute_aug])

        self.assertLess(0, len(self.model_paths))

    def oov(self, augs):
        unknown_token = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
        texts = [
            unknown_token,
            unknown_token + ' the'
        ]

        for aug in augs:
            for text in texts:
                self.assertLess(0, len(text))
                augmented_text = aug.augment(text)
                if aug.action == Action.INSERT:
                    self.assertLess(len(text.split(' ')), len(augmented_text.split(' ')))
                elif aug.action == Action.SUBSTITUTE:
                    self.assertEqual(len(text.split(' ')), len(augmented_text.split(' ')))
                else:
                    raise Exception('Augmenter is neither INSERT or SUBSTITUTE')

                self.assertTrue(aug.model.SUBWORD_PREFIX not in augmented_text)

    def insert(self, aug):
        text = 'The quick brown fox jumps over the lazy dog'

        self.assertLess(0, len(text))
        augmented_text = aug.augment(text)

        self.assertLess(len(text.split(' ')), len(augmented_text.split(' ')))
        self.assertNotEqual(text, augmented_text)
        self.assertTrue(nml.Bert.SUBWORD_PREFIX not in augmented_text)

    def substitute(self, aug):
        text = 'The quick brown fox jumps over the lazy dog'

        self.assertLess(0, len(text))
        augmented_text = aug.augment(text)

        self.assertNotEqual(text, augmented_text)
        self.assertTrue(nml.Bert.SUBWORD_PREFIX not in augmented_text)

    def substitute_stopwords(self, aug):
        texts = [
            'The quick brown fox jumps over the lazy dog'
        ]

        stopwords = [t.lower() for t in texts[0].split(' ')[:3]]
        aug.stopwords = stopwords
        aug_n = 3

        for _ in range(20):

            for text in texts:
                augmented_cnt = 0
                self.assertLess(0, len(text))

                augmented_text = aug.augment(text)
                augmented_tokens = aug.tokenizer(augmented_text)
                tokens = aug.tokenizer(text)

                for token, augmented_token in zip(tokens, augmented_tokens):
                    if token.lower() in stopwords and len(token) > aug_n:
                        self.assertEqual(token.lower(), augmented_token)
                    else:
                        augmented_cnt += 1

                self.assertGreater(augmented_cnt, 0)

    def subword(self, augs):
        # https://github.com/makcedward/nlpaug/issues/38
        text = "If I enroll in the ESPP, when will my offering begin and the price set?"

        for _ in range(100):
            for aug in augs:
                aug.augment(text)

        self.assertTrue(True)

    def not_substitute_unknown_word(self, aug):
        # https://github.com/makcedward/nlpaug/issues/38
        text = "If I enroll in the ESPP, when will my offering begin and the price set?"

        original_skip_unknown_word = aug.skip_unknown_word
        aug.skip_unknown_word = True

        for _ in range(100):
            augmented_text = aug.augment(text)
            self.assertTrue('ESPP' in augmented_text)

        aug.skip_unknown_word = original_skip_unknown_word

    def top_k_top_p(self, augs):
        text = 'The quick brown fox jumps over the lazy dog'

        for aug in augs:
            original_top_k = aug.model.top_k
            original_top_p = aug.model.top_p

            aug.model.top_k = 10000
            aug.model.top_p = 0.005

            augmented_text = aug.augment(text)

            self.assertNotEqual(text, augmented_text)
            self.assertTrue(aug.model.SUBWORD_PREFIX not in augmented_text)

            aug.model.top_k = original_top_k
            aug.model.top_p = original_top_p

    def test_incorrect_model_name(self):
        with self.assertRaises(ValueError) as error:
            naw.ContextualWordEmbsAug(model_path='unknown')

        self.assertTrue('Model name value is unexpected.' in str(error.exception))
