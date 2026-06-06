from nlpaug.augmenter.sentence.sentence_augmenter import SentenceAugmenter
from nlpaug.util import Action


class DummySentenceAugmenter(SentenceAugmenter):
    def __init__(self):
        super().__init__(action=Action.INSERT, device='cpu')

    def insert(self, data):
        return data


def test_sentence_augmenter_clean_and_duplicate_helpers():
    assert DummySentenceAugmenter.clean(' hello ') == 'hello'
    assert DummySentenceAugmenter.clean([' hello ', ' world ']) == ['hello', 'world']
    assert DummySentenceAugmenter.clean(123) == '123'
    assert DummySentenceAugmenter.is_duplicate(['a'], 'a') is True
    assert DummySentenceAugmenter.is_duplicate(['a'], 'b') is False
