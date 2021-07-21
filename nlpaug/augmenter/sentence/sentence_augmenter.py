import re

from nlpaug.util import Method
from nlpaug.util.text.tokenizer import Tokenizer
from nlpaug import Augmenter


class SentenceAugmenter(Augmenter):    
    def __init__(self, action, name='Sentence_Aug', stopwords=None, tokenizer=None, reverse_tokenizer=None,
                 device='cuda', aug_min=None, aug_max=None, aug_p=None, include_detail=False, verbose=0):
        super().__init__(
            name=name, method=Method.SENTENCE, action=action, aug_min=aug_min, aug_max=aug_max, aug_p=aug_p,
            device=device, verbose=verbose, include_detail=include_detail)
        self.tokenizer = tokenizer or Tokenizer.tokenizer
        self.reverse_tokenizer = reverse_tokenizer or Tokenizer.reverse_tokenizer
        self.stopwords = stopwords

    @classmethod
    def clean(cls, data):
        if isinstance(data, list) :
            return [d.strip() for d in data]
        return data.strip()

    @classmethod
    def is_duplicate(cls, dataset, data):
        for d in dataset:
            if d == data:
                return True
        return False
