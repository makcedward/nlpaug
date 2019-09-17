from nlpaug.util import Method
from nlpaug import Augmenter


class SentenceAugmenter(Augmenter):
    SENTENCE_SEPARATOR = '.!?'
    def __init__(self, action, name='Sentence_Aug', aug_min=1, aug_p=0.3, stopwords=None,
                 tokenizer=None, reverse_tokenizer=None, verbose=0):
        super().__init__(
            name=name, method=Method.SENTENCE, action=action, aug_min=aug_min, verbose=verbose)
        self.aug_p = aug_p
        self.tokenizer = tokenizer or self._tokenizer
        self.reverse_tokenizer = reverse_tokenizer or self._reverse_tokenizer
        self.stopwords = stopwords

    @classmethod
    def _tokenizer(cls, text):
        return text.split(' ')

    @classmethod
    def _reverse_tokenizer(cls, tokens):
        return ' '.join(tokens)

    @classmethod
    def is_duplicate(cls, dataset, data):
        for d in dataset:
            if d == data:
                return True
        return False