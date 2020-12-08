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

    def test_empty_input_for_crop(self):
        texts = ['', '           ', None]

        augs = [
            naw.RandomWordAug(action='crop',aug_p=0.5, aug_min=0)
        ]

        for aug in augs:
            for text in texts:
                augmented_text = aug.augment(text)
                self.assertTrue(augmented_text is None or augmented_text.strip() == '')

            augmented_texts = aug.augment(texts)
            for augmented_text in augmented_texts:
                self.assertTrue(augmented_text is None or augmented_text.strip() == '')

    def test_empty_input_for_insert(self):
        texts = ['', '           ']

        self.word2vec_model.action = 'insert'
        self.context_word_embs_model.action = 'insert'

        augs = [
            naw.TfIdfAug(model_path=self.tfidf_model_path, action="insert"),
            self.word2vec_model,
            self.context_word_embs_model
        ]

        for aug in augs:
            for text in texts:
                augmented_text = aug.augment(text)
                self.assertTrue(augmented_text is None or augmented_text.strip() == '')

            augmented_texts = aug.augment(texts)
            for augmented_text in augmented_texts:
                self.assertTrue(augmented_text is None or augmented_text.strip() == '')

    def test_empty_input_substitute(self):
        texts = ['', '           ']

        self.word2vec_model.action = 'substitute'
        self.context_word_embs_model.action = 'substitute'

        augs = [
            naw.SpellingAug(),
            naw.AntonymAug(),
            naw.RandomWordAug(action='substitute'),
            naw.SynonymAug(aug_src='wordnet'),
            naw.TfIdfAug(model_path=self.tfidf_model_path, action="substitute"),
            self.word2vec_model,
            self.context_word_embs_model
        ]

        for aug in augs:
            for text in texts:
                augmented_text = aug.augment(text)
                self.assertTrue(augmented_text is None or augmented_text.strip() == '')

            augmented_texts = aug.augment(texts)
            for augmented_text in augmented_texts:
                self.assertTrue(augmented_text is None or augmented_text.strip() == '')

    def test_empty_input_for_swap(self):
        texts = ['', '           ', None]
        aug = naw.RandomWordAug(action="swap")
        for text in texts:
            augmented_text = aug.augment(text)
            self.assertTrue(augmented_text is None or augmented_text.strip() == '')

        augmented_texts = aug.augment(texts)
        for augmented_text in augmented_texts:
            self.assertTrue(augmented_text is None or augmented_text.strip() == '')

    def test_empty_input_for_delete(self):
        texts = ['', '           ', None]
        augs = [
            naw.RandomWordAug(action="delete"),
            naw.RandomWordAug(action="delete", stopwords=['a', 'an', 'the'])
        ]

        for aug in augs:
            for text in texts:
                augmented_text = aug.augment(text)
                self.assertTrue(augmented_text is None or augmented_text.strip() == '')

            augmented_texts = aug.augment(texts)
            for augmented_text in augmented_texts:
                self.assertTrue(augmented_text is None or augmented_text.strip() == '')

    def test_skip_punctuation(self):
        text = '. . . . ! ? # @'

        augs = [
            # naw.ContextualWordEmbsAug(action='insert'), # After using convert_tokens_to_ids and decode function, it cannot keep it original format.
            naw.AntonymAug(),
            naw.TfIdfAug(model_path=self.tfidf_model_path, action="substitute")
        ]

        for aug in augs:
            augmented_text = aug.augment(text)
            self.assertEqual(text, augmented_text)

    def test_non_strip_input(self):
        text = ' Good boy '

        augs = [
            naw.ContextualWordEmbsAug(action='insert'),
            naw.AntonymAug(),
            naw.TfIdfAug(model_path=self.tfidf_model_path, action="substitute")
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
            naw.TfIdfAug(model_path=self.tfidf_model_path, action="substitute")
        ]

        for aug in augs:
            tokenized_text = aug.tokenizer(text)
            self.assertEqual(tokenized_text, expected_result)

    def test_multi_thread(self):
        text = 'The quick brown fox jumps over the lazy dog.'
        augs = [
            naw.RandomWordAug(),
            naw.WordEmbsAug(model_type='word2vec', model_path=self.word2vec_model_path),
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
            naw.WordEmbsAug(model_type='word2vec', model_path=self.word2vec_model_path, stopwords=stopwords)
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
            naw.WordEmbsAug(model_type='word2vec', model_path=self.word2vec_model_path,
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
        aug = naw.SpellingAug()
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

    # def test_augment_detail(self):
    #     text = 'The quick brown fox jumps over the lazy dog'
    #     augs = [
    #         naw.RandomWordAug(include_detail=True), # Delete, use SWAP later
    #         naw.ContextualWordEmbsAug(model_path='bert-base-uncased', include_detail=True) # Substitute
    #     ]

    #     for aug in augs:
    #         augmented_text, augment_details = aug.augment(text)

    #         self.assertNotEqual(text, augmented_text)
    #         self.assertGreater(len(augment_details), 0)
    #         for augment_detail in augment_details:
    #             self.assertTrue(augment_detail['orig_token'] in text)
    #             self.assertGreater(augment_detail['orig_start_pos'], -1)
    #             self.assertGreater(augment_detail['new_start_pos'], -1)
    #             self.assertGreater(augment_detail['change_seq'], 0)
    #             self.assertIn(augment_detail['action'], Action.getall())

    #         # # Get back original input by re-engineering
    #         # reengineering_text = augmented_text
    #         # for change_obj in sorted(augment_details, key=lambda item: item['orig_start_pos'], reverse=True):
    #         #     print('--------------change_obj:', change_obj)
    #         #     if change_obj['action'] == Action.DELETE:
    #         #         text_prefix = reengineering_text[:change_obj['new_start_pos']]
    #         #         text_core = ' ' + change_obj['orig_token'] + ' '
    #         #         text_suffix = reengineering_text[change_obj['new_start_pos']:]
    #         #
    #         #     elif change_obj['action'] in [Action.INSERT, Action.SUBSTITUTE]:
    #         #         text_prefix = reengineering_text[:change_obj['new_start_pos']]
    #         #         text_core = reengineering_text[change_obj['new_start_pos']:].replace(
    #         #             change_obj['new_token'], change_obj['orig_token'], 1)
    #         #         text_suffix = ''
    #         #     # TODO
    #         #     # elif change_obj['action'] in Action.SWAP:
    #         #     # TODO
    #         #     # elif change_obj['action'] in Action.ALIGN:

    #         #     print('text_prefix:', [text_prefix])
    #         #     print('text_core:', [text_core])
    #         #     print('text_suffix:', [text_suffix])
    #         #
    #         #     reengineering_text = text_prefix + text_core + text_suffix
    #         #     reengineering_text = reengineering_text.strip()
    #         #
    #         # self.assertEqual(text.lower(), reengineering_text.lower())
