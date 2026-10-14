import unittest
import os
import numpy as np
from dotenv import load_dotenv

from nlpaug.util.text.tokenizer import Tokenizer


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
