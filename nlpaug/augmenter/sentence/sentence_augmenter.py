from nlpaug.util import Method
from nlpaug import Augmenter


class SentenceAugmenter(Augmenter):
    SENTENCE_SEPARATOR = '.!?'

    def __init__(self, action, name='Sentence_Aug', stopwords=None, tokenizer=None, reverse_tokenizer=None,
                 device='cuda', verbose=0):
        super().__init__(
            name=name, method=Method.SENTENCE, action=action, aug_min=None, aug_max=None, device=device,
            verbose=verbose)
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
    def clean(cls, data):
        return data.strip()

    @classmethod
    def is_duplicate(cls, dataset, data):
        for d in dataset:
            if d == data:
                return True
        return False
