import re

from nlpaug.util import Method
from nlpaug import Augmenter


class SentenceAugmenter(Augmenter):
    SENTENCE_SEPARATOR = '.!?'
    TOKENIZER_REGEX = re.compile(r'(\W)')
    DETOKENIZER_REGEXS = [
        (re.compile(r'\s([.,:;?!%]+)([ \'"`])'), r'\1\2'), # End of sentence
        (re.compile(r'\s([.,:;?!%]+)$'), r'\1'), # End of sentence
        (re.compile(r'\s([\[\(\{\<])\s'), r' \g<1>'), # Left bracket
        (re.compile(r'\s([\]\)\}\>])\s'), r'\g<1> '), # right bracket
    ]

    def __init__(self, action, name='Sentence_Aug', stopwords=None, tokenizer=None, reverse_tokenizer=None,
                 device='cuda', include_detail=False, verbose=0):
        super().__init__(
            name=name, method=Method.SENTENCE, action=action, aug_min=None, aug_max=None, device=device,
            verbose=verbose, include_detail=include_detail)
        self.tokenizer = tokenizer or self._tokenizer
        self.reverse_tokenizer = reverse_tokenizer or self._reverse_tokenizer
        self.stopwords = stopwords

    @classmethod
    def _tokenizer(cls, text):
        tokens = cls.TOKENIZER_REGEX.split(text)
        return [t for t in tokens if len(t.strip()) > 0]

    @classmethod
    def _reverse_tokenizer(cls, tokens):
        text = ' '.join(tokens)
        for regex, sub in cls.DETOKENIZER_REGEXS:
            text = regex.sub(sub, text)
        return text.strip()

    @classmethod
    def clean(cls, data):
        return data.strip()

    @classmethod
    def is_duplicate(cls, dataset, data):
        for d in dataset:
            if d == data:
                return True
        return False
