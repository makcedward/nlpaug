import re

from nlpaug.augmenter.word.word_augmenter import WordAugmenter
from nlpaug.util import Action


class DummyWordAugmenter(WordAugmenter):
    def __init__(self, **kwargs):
        super().__init__(action=Action.SUBSTITUTE, **kwargs)

    def substitute(self, data):
        return data


def test_word_augmenter_clean_and_duplicate_helpers():
    assert DummyWordAugmenter.clean(' hello ') == 'hello'
    assert DummyWordAugmenter.clean([' hello ', None]) == ['hello', None]
    assert DummyWordAugmenter.is_duplicate(['a'], 'a') is True
    assert DummyWordAugmenter.is_duplicate(['a'], 'b') is False


def test_word_augmenter_skip_and_case_helpers():
    aug = DummyWordAugmenter(stopwords=['skip'], stopwords_regex=r'.*regex.*')

    tokens = ['hello', ',', 'skip', 'regexword', 'world']
    assert aug.pre_skip_aug(tokens) == [0, 4]
    assert aug.align_capitalization('Hello', 'world') == 'World'
    assert aug.align_capitalization('hello', 'world') == 'world'
    assert aug.get_word_case('') == 'empty'
    assert aug.get_word_case('A') == 'capitalize'
    assert aug.get_word_case('ABC') == 'upper'
    assert aug.get_word_case('abc') == 'lower'
    assert aug.get_word_case('aBc') == 'mixed'
    assert aug.get_word_case('Abc') == 'capitalize'
    assert aug.get_word_case('1abc') == 'lower'


def test_word_augmenter_index_generation_and_reserved_word_roundtrip():
    aug = DummyWordAugmenter()
    tokens = ['one', 'two', 'three', 'four']

    aug_idxes = aug._get_aug_idxes(tokens)
    assert len(aug_idxes) >= 1

    random_idxes = aug._get_random_aug_idxes(tokens)
    assert len(random_idxes) >= 1

    range_idxes = aug._get_aug_range_idxes(tokens)
    assert len(range_idxes) >= 1

    stopword_regex = re.compile(r' world ')
    replaced, reserved = aug.replace_stopword_by_reserved_word('hello world again', stopword_regex, '__RES__')
    restored = aug.replace_reserve_word_by_stopword(replaced, re.compile(r'__RES__'), reserved)

    assert '__RES__' in replaced
    assert restored.strip() == 'hello world again'
