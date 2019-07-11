from nlpaug.util import Method
from nlpaug import Augmenter
from nlpaug.util import Warning, WarningName, WarningCode, WarningMessage


class CharAugmenter(Augmenter):
    def __init__(self, action, name='Char_Aug', aug_min=1, min_char=2, aug_char_p=0.3, aug_word_p=0.3, tokenizer=None, stopwords=[],
                 verbose=0):
        super(CharAugmenter, self).__init__(
            name=name, method=Method.CHAR, action=action, aug_min=aug_min, verbose=verbose)
        self.aug_p = None
        self.aug_char_p = aug_char_p
        self.aug_word_p = aug_word_p
        self.min_char = min_char
        if tokenizer is not None:
            self.tokenizer = tokenizer
        self.stopwords = stopwords

    def tokenizer(self, text):
        return text.split(' ')

    def token2char(self, word):
        return list(word)

    def reverse_tokenizer(self, tokens):
        return ' '.join(tokens)

    def skip_aug(self, token_idxes, tokens):
        return token_idxes

    def _get_aug_idxes(self, tokens, aug_p, mode):
        if mode == Method.CHAR:
            if len(tokens) <= self.min_char:
                return None

        aug_cnt = self.generate_aug_cnt(len(tokens), aug_p)
        idxes = [i for i, t in enumerate(tokens)]
        if mode == Method.WORD:
            idxes = [i for i in idxes if tokens[i] not in self.stopwords]
        elif mode == Method.CHAR:
            idxes = self.skip_aug(idxes, tokens)

        if len(idxes) == 0:
            if self.verbose > 0:
                exception = Warning(name=WarningName.OUT_OF_VOCABULARY,
                                    code=WarningCode.WARNING_CODE_002, msg=WarningMessage.NO_WORD)
                exception.output()
            return None
        if len(idxes) < aug_cnt:
            aug_cnt = len(idxes)
        aug_idexes = self.sample(idxes, aug_cnt)
        return aug_idexes

