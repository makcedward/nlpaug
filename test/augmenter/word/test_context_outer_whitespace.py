import unittest
from types import SimpleNamespace

from nlpaug.augmenter.word.context_word_embs import (
    ContextualWordEmbsAug,
    split_outer_whitespace,
)


class FakeTokenizer:
    def tokenize(self, text):
        return text.split(' ') if text else []

    def convert_tokens_to_ids(self, tokens):
        return list(range(len(tokens)))

    def convert_tokens_to_string(self, tokens):
        return ' '.join(tokens)

    def decode(self, ids):
        # Simulate HF decode collapsing whitespace to single spaces
        return 'decoded'

    def convert_ids_to_tokens(self, ids):
        return ['t']


class FakeModel:
    UNKNOWN_TOKEN = '[UNK]'

    def __init__(self, decoded='hello FAKE world'):
        self._decoded = decoded
        self.tokenizer = FakeTokenizer()

    def get_tokenizer(self):
        tok = self.tokenizer
        orig_decode = tok.decode
        decoded = self._decoded

        class T:
            def tokenize(self, text):
                return tok.tokenize(text)

            def convert_tokens_to_ids(self, tokens):
                return tok.convert_tokens_to_ids(tokens)

            def convert_tokens_to_string(self, tokens):
                return tok.convert_tokens_to_string(tokens)

            def decode(self, ids):
                return decoded

            def convert_ids_to_tokens(self, ids):
                return tok.convert_ids_to_tokens(ids)

        return T()

    def get_mask_token(self):
        return '[MASK]'

    def get_subword_prefix(self):
        return '##'

    def get_unknown_token(self):
        return '[UNK]'

    def get_max_num_token(self):
        return 512

    def get_model(self):
        return SimpleNamespace(config=SimpleNamespace(max_position_embeddings=-1))

    def predict(self, masked_texts, target_words=None, n=2):
        return [['FAKE'] for _ in masked_texts]


def make_aug(decoded='hello FAKE world'):
    aug = object.__new__(ContextualWordEmbsAug)
    aug.model_type = 'bert'
    aug.model_path = 'bert-base-cased'
    aug.model = FakeModel(decoded=decoded)
    aug.action = 'substitute'
    aug.aug_min = 1
    aug.aug_max = 1
    aug.aug_p = 1.0
    aug.stopwords = None
    aug.stopwords_regex = None
    aug.stopword_reg = None
    aug.reserve_word_reg = None
    aug.max_num_token = 512
    aug.parent_change_seq = 0
    aug.include_detail = False
    aug.verbose = 0
    aug.device = 'cpu'
    return aug


class TestSplitOuterWhitespace(unittest.TestCase):
    def test_spaces(self):
        self.assertEqual(('  ', 'hello', ' '), split_outer_whitespace('  hello '))

    def test_tabs_newlines_crlf(self):
        self.assertEqual(('\t', 'a', '\n'), split_outer_whitespace('\ta\n'))
        leading, core, trailing = split_outer_whitespace('\r\n  hello world  \r\n')
        self.assertEqual('\r\n  ', leading)
        self.assertEqual('hello world', core)
        self.assertEqual('  \r\n', trailing)

    def test_no_whitespace(self):
        self.assertEqual(('', 'abc', ''), split_outer_whitespace('abc'))

    def test_empty_and_whitespace_only(self):
        self.assertEqual(('', '', ''), split_outer_whitespace(''))
        # whitespace-only: key property is round-trip preservation
        for ws in ['   ', '\n\n', ' \t ']:
            leading, core, trailing = split_outer_whitespace(ws)
            self.assertEqual(ws, leading + core + trailing)

    def test_round_trip(self):
        for text in ['  hello world  ', '\n  indented\nline  \n', 'x', '', '   ']:
            leading, core, trailing = split_outer_whitespace(text)
            self.assertEqual(text, leading + core + trailing)


class TestPreserveOuterWhitespace(unittest.TestCase):
    def test_substitute_preserves_outer_whitespace(self):
        aug = make_aug(decoded='hello FAKE world')
        out = aug.substitute('  hello world  ')
        self.assertTrue(out.startswith('  '), out)
        self.assertTrue(out.endswith('  '), out)
        self.assertIn('FAKE', out)

    def test_insert_preserves_outer_whitespace(self):
        aug = make_aug(decoded='hello FAKE world')
        aug.action = 'insert'
        out = aug.insert('\nhello world\n')
        self.assertTrue(out.startswith('\n'), repr(out))
        self.assertTrue(out.endswith('\n'), repr(out))

    def test_poem_indentation_outer_edges(self):
        # From #323: outer leading/trailing newlines+spaces must survive
        text = '\n         hello world\n         '
        aug = make_aug(decoded='hello FAKE world')
        out = aug.substitute(text)
        self.assertTrue(out.startswith('\n         '), repr(out))
        self.assertTrue(out.endswith('\n         '), repr(out))


if __name__ == '__main__':
    unittest.main()
