import unittest
import os
from dotenv import load_dotenv

import nlpaug.augmenter.word as naw
from nlpaug.util import Action, Doc


class TestWord(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env_config_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', '.env'))
        load_dotenv(env_config_path)

        cls.word2vec_model_path = os.path.join(os.environ.get("MODEL_DIR"), 'word', 'word_embs', 'GoogleNews-vectors-negative300.bin')
        cls.word2vec_model = naw.WordEmbsAug(model_type='word2vec', model_path=cls.word2vec_model_path)
        cls.context_word_embs_model = naw.ContextualWordEmbsAug()

        cls.tfidf_model_path = os.path.join(os.environ.get("MODEL_DIR"), 'word', 'tfidf')

        cls._train_tfidf(cls)

    @classmethod
    def tearDownClass(self):
        os.remove(os.path.join(self.tfidf_model_path, 'tfidfaug_w2idf.txt'))
        os.remove(os.path.join(self.tfidf_model_path, 'tfidfaug_w2tfidf.txt'))

    def _train_tfidf(self):
        import sklearn.datasets
        import re
        import nlpaug.model.word_stats as nmw

        def _tokenizer(text, token_pattern=r"(?u)\b\w\w+\b"):
            token_pattern = re.compile(token_pattern)
            return token_pattern.findall(text)

        # Load sample data
        train_data = sklearn.datasets.fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
        train_x = train_data.data

        # Tokenize input
        train_x_tokens = [_tokenizer(x) for x in train_x]

        # Train TF-IDF model
        if not os.path.exists(self.tfidf_model_path):
            os.makedirs(self.tfidf_model_path)

        tfidf_model = nmw.TfIdf()
        tfidf_model.train(train_x_tokens)
        tfidf_model.save(self.tfidf_model_path)

    # def test_empty_input_for_crop(self):
    #     texts = ['', '           ', None]

    #     augs = [
    #         naw.RandomWordAug(action='crop',aug_p=0.5, aug_min=0)
    #     ]

    #     for aug in augs:
    #         for text in texts:
    #             augmented_data = aug.augment(text)
    #             self.assertTrue(len(augmented_data) == 0 or augmented_data[0].strip() == '')

    #         augmented_texts = aug.augment(texts)
    #         for augmented_text in augmented_texts:
    #             self.assertTrue(augmented_text is None or augmented_text.strip() == '')

    # def test_empty_input_for_insert(self):
    #     texts = ['', '           ']

    #     self.word2vec_model.action = 'insert'
    #     self.context_word_embs_model.action = 'insert'

    #     augs = [
    #         naw.TfIdfAug(model_path=self.tfidf_model_path, action="insert"),
    #         self.word2vec_model,
    #         self.context_word_embs_model
    #     ]

    #     for aug in augs:
    #         for text in texts:
    #             augmented_data = aug.augment(text)
    #             self.assertTrue(len(augmented_data) == 0 or augmented_data[0].strip() == '')

    #         augmented_data = aug.augment(texts)
    #         for augmented_text in augmented_data:
    #             self.assertTrue(augmented_text is None or augmented_text.strip() == '')

    # def test_empty_input_substitute(self):
    #     texts = ['', '           ']

    #     self.word2vec_model.action = 'substitute'
    #     self.context_word_embs_model.action = 'substitute'

    #     augs = [
    #         naw.SpellingAug(),
    #         naw.AntonymAug(),
    #         naw.RandomWordAug(action='substitute'),
    #         naw.SynonymAug(aug_src='wordnet'),
    #         naw.TfIdfAug(model_path=self.tfidf_model_path, action="substitute"),
    #         self.word2vec_model,
    #         self.context_word_embs_model
    #     ]

    #     for aug in augs:
    #         for text in texts:
    #             augmented_data = aug.augment(text)
    #             self.assertTrue(len(augmented_data) == 0 or augmented_data[0].strip() == '')

    #         augmented_data = aug.augment(texts)
    #         for augmented_text in augmented_data:
    #             self.assertTrue(augmented_text is None or augmented_text.strip() == '')

    # def test_empty_input_for_swap(self):
    #     texts = ['', '           ', None]
    #     aug = naw.RandomWordAug(action="swap")
    #     for text in texts:
    #         augmented_data = aug.augment(text)
    #         self.assertTrue(len(augmented_data) == 0 or augmented_data[0].strip() == '')

    #     augmented_data = aug.augment(texts)
    #     for augmented_text in augmented_data:
    #         self.assertTrue(augmented_text is None or augmented_text.strip() == '')

    # def test_empty_input_for_delete(self):
    #     texts = ['', '           ', None]
    #     augs = [
    #         naw.RandomWordAug(action="delete"),
    #         naw.RandomWordAug(action="delete", stopwords=['a', 'an', 'the'])
    #     ]

    #     for aug in augs:
    #         for text in texts:
    #             augmented_data = aug.augment(text)
    #             self.assertTrue(len(augmented_data) == 0 or augmented_data[0].strip() == '')

    #         augmented_data = aug.augment(texts)
    #         for augmented_text in augmented_data:
    #             self.assertTrue(augmented_text is None or augmented_text.strip() == '')

    # def test_skip_punctuation(self):
    #     text = '. . . . ! ? # @'

    #     augs = [
    #         # naw.ContextualWordEmbsAug(action='insert'), # After using convert_tokens_to_ids and decode function, it cannot keep it original format.
    #         naw.AntonymAug(),
    #         naw.TfIdfAug(model_path=self.tfidf_model_path, action="substitute")
    #     ]

    #     for aug in augs:
    #         augmented_data = aug.augment(text)
    #         augmented_text = augmented_data[0]
    #         self.assertEqual(text, augmented_text)

    # def test_non_strip_input(self):
    #     text = ' Good boy '

    #     augs = [
    #         naw.ContextualWordEmbsAug(action='insert'),
    #         naw.AntonymAug(),
    #         naw.TfIdfAug(model_path=self.tfidf_model_path, action="substitute")
    #     ]

    #     for aug in augs:
    #         augmented_data = aug.augment(text)
    #         augmented_text = augmented_data[0]
    #         self.assertNotEqual(text, augmented_text)

    # def test_excessive_space(self):
    #     # https://github.com/makcedward/nlpaug/issues/48
    #     text = 'The  quick brown fox        jumps over the lazy dog . 1  2 '
    #     expected_result = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '.', '1', '2']

    #     augs = [
    #         naw.ContextualWordEmbsAug(action='insert'),
    #         naw.AntonymAug(),
    #         naw.TfIdfAug(model_path=self.tfidf_model_path, action="substitute")
    #     ]

    #     for aug in augs:
    #         tokenized_text = aug.tokenizer(text)
    #         self.assertEqual(tokenized_text, expected_result)

    # def test_multi_thread(self):
    #     text = 'The quick brown fox jumps over the lazy dog.'
    #     augs = [
    #         naw.RandomWordAug(),
    #         naw.WordEmbsAug(model_type='word2vec', model_path=self.word2vec_model_path),
    #         naw.ContextualWordEmbsAug(
    #             model_path='distilroberta-base', action="substitute", device='cpu')
    #     ]

    #     for num_thread in [1, 3]:
    #         for aug in augs:
    #             augmented_data = aug.augment(text, n=num_thread, num_thread=num_thread)
    #             self.assertEqual(len(augmented_data), num_thread)

    # def test_stopwords(self):
    #     text = 'The quick brown fox jumps over the lazy dog.'
    #     stopwords = ['The', 'brown', 'fox', 'jumps', 'the', 'dog']

    #     augs = [
    #         naw.RandomWordAug(action="delete", stopwords=stopwords),
    #         naw.ContextualWordEmbsAug(stopwords=stopwords),
    #         naw.WordEmbsAug(model_type='word2vec', model_path=self.word2vec_model_path, stopwords=stopwords)
    #     ]

    #     for aug in augs:
    #         for i in range(10):
    #             augmented_data = aug.augment(text)
    #             augmented_text = augmented_data[0]
    #             self.assertTrue(
    #                 'quick' not in augmented_text or 'over' not in augmented_text or 'lazy' not in augmented_text)

    # # https://github.com/makcedward/nlpaug/issues/247
    # def test_stopword_for_preprocess(self):
    #     stopwords = ["[id]", "[year]"]
    #     texts = [
    #         "My id is [id], and I born in [year]", # with stopwords as last word
    #         "[id] id is [id], and I born in [year]", # with stopwords as first word
    #         "[id] [id] Id is [year] [id]", # continuous stopwords
    #         "[id]  [id] Id is [year]   [id]", # continuous stopwords with space
    #         "My id is [id], and I   [id] born in [year] a[year] [year]b aa[year]", # with similar stopwords
    #         "My id is [id], and I born [UNK] [year]", # already have reserved word. NOT handling now
    #     ]
    #     expected_replaced_texts = [
    #         'My id is [UNK], and I born in [UNK]',
    #         '[UNK] id is [UNK], and I born in [UNK]',
    #         '[UNK] [UNK] Id is [UNK] [UNK]',
    #         '[UNK]  [UNK] Id is [UNK]   [UNK]',
    #         'My id is [UNK], and I   [UNK] born in [UNK] a[year] [year]b aa[year]',
    #         "My id is [UNK], and I born [UNK] [UNK]",
    #     ]
    #     expected_reserved_tokens = [
    #         ['[year]', '[id]'],
    #         ['[year]', '[id]', '[id]'],
    #         ['[id]', '[year]', '[id]', '[id]'],
    #         ['[id]', '[year]', '[id]', '[id]'],
    #         ['[year]', '[id]', '[id]'],
    #         ['[year]', '[id]']
    #     ]
    #     expected_reversed_texts = [
    #         'My id is [id], and I born in [year]',
    #         '[id] id is [id], and I born in [year]',
    #         '[id] [id] Id is [year] [id]',
    #         '[id]  [id] Id is [year]   [id]',
    #         'My id is [id], and I   [id] born in [year] a[year] [year]b aa[year]',
    #         'My id is [UNK], and I born [id] [year]'
    #     ]

    #     augs = [
    #         naw.ContextualWordEmbsAug(
    #             model_path='bert-base-uncased', action="insert", stopwords=stopwords),
    #         naw.ContextualWordEmbsAug(
    #             model_path='bert-base-uncased', action="substitute", stopwords=stopwords)
    #     ]
        
    #     for aug in augs:
    #         unknown_token = aug.model.get_unknown_token() or aug.model.UNKNOWN_TOKEN

    #         for expected_text, expected_reserved_token_list, expected_reversed_text, text in zip(
    #             expected_replaced_texts, expected_reserved_tokens, expected_reversed_texts, texts):
    #             replaced_text, reserved_stopwords = aug.replace_stopword_by_reserved_word(
    #                 text, aug.stopword_reg, unknown_token)
    #             assert expected_text == replaced_text
    #             assert expected_reserved_token_list == reserved_stopwords
                
    #             reversed_text = aug.replace_reserve_word_by_stopword(
    #                 replaced_text, aug.reserve_word_reg, reserved_stopwords)
    #             assert expected_reversed_text == reversed_text    

    # # https://github.com/makcedward/nlpaug/issues/81
    # def test_stopwords_regex(self):
    #     text = 'The quick brown fox jumps over the lazy dog.'
    #     stopwords_regex = "( [a-zA-Z]{1}ox | [a-z]{1}og|(brown)|[a-zA-z]{1}he)|[a-z]{2}mps "

    #     augs = [
    #         naw.RandomWordAug(action="delete", stopwords_regex=stopwords_regex),
    #         naw.ContextualWordEmbsAug(stopwords_regex=stopwords_regex),
    #         naw.WordEmbsAug(model_type='word2vec', model_path=self.word2vec_model_path,
    #                         stopwords_regex=stopwords_regex)
    #     ]

    #     for aug in augs:
    #         for i in range(10):
    #             augmented_data = aug.augment(text)
    #             augmented_text = augmented_data[0]
    #             self.assertTrue(
    #                 'quick' not in augmented_text or 'over' not in augmented_text or 'lazy' not in augmented_text)

    # # https://github.com/makcedward/nlpaug/issues/82
    def test_case(self):
        # Swap
        aug = naw.RandomWordAug(action='swap')
        augmented_data = aug.augment('aA bB')
        augmented_text = augmented_data[0]
        self.assertEqual('bB aA', augmented_text)

        data = 'I love McDonalds'
        doc = Doc(data, aug.tokenizer(data))
        augmented_tokens = aug.change_case(doc, 1, 0, 1).get_augmented_tokens()
        self.assertEqual(['Love', 'I', 'McDonalds'], augmented_tokens)
        doc = Doc(data, aug.tokenizer(data))
        augmented_tokens = aug.change_case(doc, 0, 1, 1).get_augmented_tokens()
        self.assertEqual(['Love', 'I', 'McDonalds'], augmented_tokens)

        data = 'He loves McDonalds'
        doc = Doc(data, aug.tokenizer(data))
        augmented_tokens = aug.change_case(doc, 1, 0, 1).get_augmented_tokens()
        self.assertEqual(['Loves', 'he', 'McDonalds'], augmented_tokens)
        doc = Doc(data, aug.tokenizer(data))
        augmented_tokens = aug.change_case(doc, 0, 1, 1).get_augmented_tokens()
        self.assertEqual(['Loves', 'he', 'McDonalds'], augmented_tokens)
        doc = Doc(data, aug.tokenizer(data))
        augmented_tokens = aug.change_case(doc, 2, 1, 1).get_augmented_tokens()
        self.assertEqual(['He', 'McDonalds', 'loves'], augmented_tokens)

        # Insert
        aug = naw.TfIdfAug(model_path=self.tfidf_model_path, action='insert')
        expected = False
        for i in range(10):
            augmented_data = aug.augment('Good')
            augmented_text = augmented_data[0]
            if 'good' in augmented_text and aug.get_word_case(augmented_text.split(' ')[0]) == 'capitalize':
                expected = True
                break
        self.assertTrue(expected)

        # Substitute
        aug = naw.RandomWordAug(action='substitute', target_words=['abc'])
        expected = False
        for i in range(10):
            augmented_data = aug.augment('I love')
            augmented_text = augmented_data[0]
            if augmented_text == 'Abc love':
                expected = True
                break
        self.assertTrue(expected)

        aug = naw.AntonymAug()
        aug_data = aug.augment('Happy')
        self.assertEqual('Unhappy', aug_data[0])

        # Do not change if target word is non-lower
        aug = naw.SpellingAug()
        aug_data = aug.augment('Re')
        self.assertEqual('RE', aug_data[0])

        # Delete case
        aug = naw.RandomWordAug(action='delete')
        expected = False
        for i in range(10):
            augmented_data = aug.augment('I love')
            augmented_text = augmented_data[0]
            if augmented_text == 'Love':
                expected = True
                break
        self.assertTrue(expected)
