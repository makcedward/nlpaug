import unittest
import os
import numpy as np
from dotenv import load_dotenv

from nlpaug.util.text.tokenizer import Tokenizer, split_sentences, split_long_document


class TestTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env_config_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '.env'))
        load_dotenv(env_config_path)

    def test_tokenizer(self):
        text = 'The quick brown fox jumps over the lazy dog?'

        tokens = Tokenizer.tokenizer(text)
        expected_tokens = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '?']
        self.assertEqual(tokens, expected_tokens)
        

    def test_reverse_tokenizer(self):
        text = 'The quick (brown) [fox] {jumps} over the lazy dog?'

        tokens = Tokenizer.tokenizer(text)
        self.assertEqual(text, Tokenizer.reverse_tokenizer(tokens))

    def test_reverse_tokenizer_with_original_preserves_separators(self):
        text = "un espace\u00a0avant d'interrogation ?"
        tokens = Tokenizer.tokenizer(text)
        tokens[tokens.index('espace')] = 'exemple'

        result = Tokenizer.reverse_tokenizer_with_original(tokens, text)

        self.assertEqual("un exemple\u00a0avant d'interrogation ?", result)

    def test_split_sentences_for_long_documents(self):
        # Issue #336: long documents need sentence-level chunking
        text = 'Hello world. This is a test! Is it working? Yes.'
        self.assertEqual(
            split_sentences(text),
            ['Hello world.', 'This is a test!', 'Is it working?', 'Yes.'])
        self.assertEqual(split_sentences(''), [])
        self.assertEqual(split_sentences('   '), [])
        self.assertEqual(split_sentences('No boundary'), ['No boundary'])

    def test_split_long_document_chunks(self):
        text = 'A. B. C. D. E. F.'
        self.assertEqual(
            split_long_document(text, max_sentences=2),
            ['A. B.', 'C. D.', 'E. F.'])
        self.assertEqual(split_long_document('', max_sentences=2), [])
        with self.assertRaises(ValueError):
            split_long_document(text, max_sentences=0)
        # token cap splits single-boundary long text
        long_text = ' '.join(['word'] * 250)
        chunks = split_long_document(long_text, max_sentences=5, max_tokens=100)
        self.assertEqual(len(chunks), 3)
        self.assertTrue(all(len(c.split()) <= 100 for c in chunks))

    def test_reverse_tokenizer_preserves_chinese_issue_231(self):
        tokens = Tokenizer.tokenizer('咋就不行了？')
        self.assertEqual(Tokenizer.reverse_tokenizer(tokens), '咋就不行了？')
        self.assertEqual(
            Tokenizer.reverse_tokenizer(['咋', '就', '不', '行', '了', '？']),
            '咋就不行了？')

    def test_split_long_document_multi_sentence_token_cap(self):
        text = 'First sentence here. Second sentence here. Third one here. Fourth here.'
        chunks = split_long_document(text, max_sentences=10, max_tokens=5)
        self.assertTrue(all(len(c.split()) <= 5 for c in chunks))
        self.assertEqual(' '.join(chunks).split(), text.split())
        with self.assertRaises(ValueError):
            split_long_document(text, max_sentences=2, max_tokens=0)

    def test_add_space_around_punctuation(self):
        from nlpaug.util.text.tokenizer import add_space_around_punctuation, split_sentence
        self.assertIn(' ', add_space_around_punctuation('hello,world'))
        self.assertTrue(len(split_sentence('hello world')) >= 1)
