import unittest

from nlpaug.augmenter.word.context_word_embs import ContextualWordEmbsAug


class FakeTokenizer:
    def __init__(self, ret):
        self._ret = ret

    def convert_tokens_to_string(self, tokens):
        if isinstance(self._ret, Exception):
            raise self._ret
        return self._ret

    def convert_tokens_to_ids(self, tokens):
        return [1, 2]

    def decode(self, ids):
        return ' fallback text '


class FakeModel:
    def __init__(self, ret):
        self._tok = FakeTokenizer(ret)

    def get_tokenizer(self):
        return self._tok


def make_aug(ret):
    aug = ContextualWordEmbsAug.__new__(ContextualWordEmbsAug)
    aug.model = FakeModel(ret)
    return aug


class TestTokensToStringCompat(unittest.TestCase):
    """Regression test for issue #284.

    Some tokenizers return a list from convert_tokens_to_string instead of
    a str, raising AttributeError: 'list' object has no attribute 'strip'.
    """

    def test_str_passthrough(self):
        self.assertEqual('hello world', make_aug('hello world  ')._tokens_to_string(['x']))

    def test_singleton_list(self):
        # Suggested fix in #284 was `[0].strip()`; must work but not break str path
        self.assertEqual('hello world', make_aug(['hello world  '])._tokens_to_string(['x']))

    def test_multi_list_joined(self):
        self.assertEqual('hello world', make_aug(['hello', 'world'])._tokens_to_string(['x']))

    def test_empty_list(self):
        self.assertEqual('', make_aug([])._tokens_to_string(['x']))

    def test_exception_falls_back_to_decode(self):
        self.assertEqual('fallback text', make_aug(RuntimeError('boom'))._tokens_to_string(['x']))


if __name__ == '__main__':
    unittest.main()
