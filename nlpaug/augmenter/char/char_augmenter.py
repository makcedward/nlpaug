from nlpaug.util import Method
from nlpaug import Augmenter


class CharAugmenter(Augmenter):
    def __init__(self, action, name='Char_Aug', aug_min=1, min_char=2, aug_p=0.3, tokenizer=None):
        super(CharAugmenter, self).__init__(
            name=name, method=Method.CHAR, action=action, aug_min=aug_min)
        self.aug_p = aug_p
        self.min_char = min_char
        if tokenizer is not None:
            self.tokenizer = tokenizer

    def tokenizer(self, text):
        return text.split(' ')

    def token2char(self, word):
        return list(word)

    def reverse_tokenizer(self, tokens):
        return ' '.join(tokens)
