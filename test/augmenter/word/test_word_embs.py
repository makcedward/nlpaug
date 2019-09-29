import unittest
import os
from dotenv import load_dotenv

import nlpaug.augmenter.word as naw


class TestWordEmbsAug(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env_config_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', '.env'))
        load_dotenv(env_config_path)

        model_dir = os.environ.get("MODEL_DIR")

        full_test_case = False

        cls.augs = [
            naw.WordEmbsAug(model_type='word2vec', model_path=model_dir+'GoogleNews-vectors-negative300.bin'),
            naw.WordEmbsAug(model_type='glove', model_path=model_dir+'glove.6B.50d.txt'),
            naw.WordEmbsAug(model_type='fasttext', model_path=model_dir + 'wiki-news-300d-1M.vec')
        ]

        if full_test_case:
            cls.augs.extend([
                naw.WordEmbsAug(model_type='glove', model_path=model_dir+'glove.42B.300d.txt'),
                naw.WordEmbsAug(model_type='glove', model_path=model_dir+'glove.840B.300d.txt'),
                naw.WordEmbsAug(model_type='glove', model_path=model_dir+'glove.twitter.27B.25d.txt'),
                naw.WordEmbsAug(model_type='glove', model_path=model_dir+'glove.twitter.27B.50d.txt'),
                naw.WordEmbsAug(model_type='glove', model_path=model_dir+'glove.twitter.27B.100d.txt'),
                naw.WordEmbsAug(model_type='glove', model_path=model_dir+'glove.twitter.27B.200d.txt'),
                naw.WordEmbsAug(model_type='fasttext', model_path=model_dir+'wiki-news-300d-1M-subword.vec'),
                naw.WordEmbsAug(model_type='fasttext', model_path=model_dir+'crawl-300d-2M.vec'),
                naw.WordEmbsAug(model_type='fasttext', model_path=model_dir+'crawl-300d-2M-subword.vec'),
            ])

    def test_oov(self):
        unknown_token = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'

        for aug in self.augs:
            aug.action = 'substitute'

            augmented_text = aug.augment(unknown_token)
            self.assertEqual(unknown_token, augmented_text)

            text = unknown_token + ' the'

            augmented_text = aug.augment(text)
            self.assertNotEqual(text, augmented_text)
            self.assertTrue(unknown_token in augmented_text)

    def test_insert(self):
        texts = [
            'The quick brown fox jumps over the lazy dog'
        ]

        for aug in self.augs:
            aug.action = 'insert'

            for text in texts:
                self.assertLess(0, len(text))
                augmented_text = aug.augment(text)

                self.assertLess(len(text.split(' ')), len(augmented_text.split(' ')))
                self.assertNotEqual(text, augmented_text)

        self.assertLess(0, len(texts))

    def test_substitute(self):
        texts = [
            'The quick brown fox jumps over the lazy dog'
        ]

        for aug in self.augs:
            aug.action = 'substitute'

            for text in texts:
                self.assertLess(0, len(text))
                augmented_text = aug.augment(text)

                self.assertNotEqual(text, augmented_text)

        self.assertLess(0, len(texts))

    def test_bogus_fasttext_loading(self):
        import nlpaug.model.word_embs.fasttext as ft
        test_file = os.path.join(os.path.dirname(__file__), 'bogus_fasttext.vec')
        expected_vector = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        fasttext = ft.Fasttext()
        fasttext.read(test_file)

        for word in fasttext.w2v:
            self.assertSequenceEqual(list(fasttext.w2v[word]), expected_vector)

        self.assertSequenceEqual(["test1", "test2", "test_3", "test 4", "test -> 5"], fasttext.vocab)

        self.assertEqual(len(fasttext.vectors), 5)

    def test_incorrect_model_type(self):
        with self.assertRaises(ValueError) as error:
            naw.WordEmbsAug(
                model_type='test_model_type',
                model_path=os.environ.get("MODEL_DIR") + 'GoogleNews-vectors-negative300.bin')

        self.assertTrue('Model type value is unexpected.' in str(error.exception))

    def test_reset_top_k(self):
        original_aug = naw.WordEmbsAug(
            model_type='word2vec', model_path=os.environ.get("MODEL_DIR") + 'GoogleNews-vectors-negative300.bin')
        original_top_k = original_aug.model.top_k

        new_aug = naw.WordEmbsAug(
            model_type='word2vec', model_path=os.environ.get("MODEL_DIR") + 'GoogleNews-vectors-negative300.bin',
            top_k=original_top_k+1)
        new_top_k = new_aug.model.top_k

        self.assertEqual(original_top_k+1, new_top_k)

    def test_case_insensitive(self):
        retry_cnt = 10

        text = 'Good'
        aug = naw.WordEmbsAug(
            model_type='word2vec', model_path=os.environ.get("MODEL_DIR") + 'GoogleNews-vectors-negative300.bin',
            top_k=2)

        for _ in range(retry_cnt):
            augmented_text = aug.augment(text)
            self.assertNotEqual(text.lower(), augmented_text.lower())

        self.assertLess(0, retry_cnt)