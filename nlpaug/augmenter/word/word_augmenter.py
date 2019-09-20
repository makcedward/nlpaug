from nlpaug.util import Method
from nlpaug import Augmenter
from nlpaug.util import WarningException, WarningName, WarningCode, WarningMessage


class WordAugmenter(Augmenter):
    def __init__(self, action, name='Word_Aug', aug_min=1, aug_p=0.3, stopwords=None,
                 tokenizer=None, reverse_tokenizer=None, verbose=0):
        super().__init__(
            name=name, method=Method.WORD, action=action, aug_min=aug_min, verbose=verbose)
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

    def skip_aug(self, token_idxes, tokens):
        return token_idxes

    @classmethod
    def is_duplicate(cls, dataset, data):
        for d in dataset:
            if d == data:
                return True
        return False

    @classmethod
    def align_capitalization(cls, src_token, dest_token):
        # For whole word is upper case
        if src_token[0].isupper() and len(src_token) > 1 and src_token[1].isupper():
            return dest_token.upper()
        # For capitalize word
        elif src_token[0].isupper():
            return dest_token.capitalize()
        else:
            return dest_token

    def _get_aug_idxes(self, tokens):
        aug_cnt = self.generate_aug_cnt(len(tokens))
        word_idxes = [i for i, t in enumerate(tokens) if self.stopwords is None or t not in self.stopwords]
        word_idxes = self.skip_aug(word_idxes, tokens)
        if len(word_idxes) == 0:
            if self.verbose > 0:
                exception = WarningException(name=WarningName.OUT_OF_VOCABULARY,
                                             code=WarningCode.WARNING_CODE_002, msg=WarningMessage.NO_WORD)
                exception.output()
            return None
        if len(word_idxes) < aug_cnt:
            aug_cnt = len(word_idxes)
        aug_idexes = self.sample(word_idxes, aug_cnt)
        return aug_idexes

    def _get_random_aug_idxes(self, tokens):
        aug_cnt = self.generate_aug_cnt(len(tokens))
        word_idxes = [i for i in range(len(tokens))]
        if len(word_idxes) < aug_cnt:
            aug_cnt = len(word_idxes)

        aug_idxes = self.sample(word_idxes, aug_cnt)

        return aug_idxes
