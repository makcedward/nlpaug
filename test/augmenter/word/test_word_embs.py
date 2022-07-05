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

        cls.text = 'The quick brown fox jumps over the lazy dog.'

        cls.word2vec_model_path = os.path.join(model_dir, 'word', 'word_embs', 'GoogleNews-vectors-negative300.bin')

        cls.augs = [
            naw.WordEmbsAug(model_type='word2vec', model_path=cls.word2vec_model_path),
            naw.WordEmbsAug(model_type='glove', model_path=os.path.join(model_dir, 'word', 'word_embs', 'glove.6B.50d.txt')),
            naw.WordEmbsAug(model_type='fasttext', model_path=os.path.join(model_dir, 'word', 'word_embs', 'wiki-news-300d-1M.vec'))
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

            augmented_data = aug.augment(unknown_token)
            augmented_text = augmented_data[0]
            self.assertEqual(unknown_token, augmented_text)

            text = unknown_token + ' the'

            augmented_data = aug.augment(text)
            augmented_text = augmented_data[0]
            self.assertNotEqual(text, augmented_text)
            self.assertTrue(unknown_token in augmented_text)

    def test_insert(self):
        for aug in self.augs:
            aug.action = 'insert'

            self.assertLess(0, len(self.text))
            augmented_data = aug.augment(self.text)
            augmented_text = augmented_data[0]

            self.assertLess(len(self.text.split(' ')), len(augmented_text.split(' ')))
            self.assertNotEqual(self.text, augmented_text)

    def test_substitute(self):
        for aug in self.augs:
            aug.action = 'substitute'

            self.assertLess(0, len(self.text))
            augmented_data = aug.augment(self.text)
            augmented_text = augmented_data[0]
            self.assertNotEqual(self.text, augmented_text)

    def test_incorrect_model_type(self):
        with self.assertRaises(ValueError) as error:
            naw.WordEmbsAug(
                model_type='test_model_type',
                model_path=self.word2vec_model_path)

        self.assertTrue('Model type value is unexpected.' in str(error.exception))

    def test_reset_top_k(self):
        original_aug = naw.WordEmbsAug(
            model_type='word2vec', model_path=self.word2vec_model_path)
        original_top_k = original_aug.model.top_k

        new_aug = naw.WordEmbsAug(
            model_type='word2vec', model_path=self.word2vec_model_path,
            top_k=original_top_k+1)
        new_top_k = new_aug.model.top_k

        self.assertEqual(original_top_k+1, new_top_k)

    def test_case_insensitive(self):
        retry_cnt = 10

        text = 'Good'
        aug = naw.WordEmbsAug(
            model_type='word2vec', model_path=self.word2vec_model_path,
            top_k=2)

        for _ in range(retry_cnt):
            augmented_data = aug.augment(text)
            augmented_text = augmented_data[0]
            self.assertNotEqual(text.lower(), augmented_text.lower())

        self.assertLess(0, retry_cnt)
