from nlpaug.util import Method
from nlpaug import Augmenter
from nlpaug.util import Warning, WarningName, WarningCode, WarningMessage


class WordAugmenter(Augmenter):
    def __init__(self, action, name='Word_Aug', aug_min=1, aug_p=0.3, tokenizer=None, stopwords=[], verbose=0):
        super(WordAugmenter, self).__init__(
            name=name, method=Method.WORD, action=action, aug_min=aug_min, verbose=verbose)
        self.aug_p = aug_p
        if tokenizer is not None:
            self.tokenizer = tokenizer
        self.stopwords = stopwords
        
    def tokenizer(self, text):
        return text.split(' ')

    def reverse_tokenizer(self, tokens):
        return ' '.join(tokens)

    def skip_aug(self, token_idxes, tokens):
        return token_idxes

    def align_capitalization(self, src_token, dest_token):
        """
            Simulate capitalized string
        """
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
        word_idxes = [i for i, t in enumerate(tokens) if t not in self.stopwords]
        word_idxes = self.skip_aug(word_idxes, tokens)
        if len(word_idxes) == 0:
            if self.verbose > 0:
                exception = Warning(name=WarningName.OUT_OF_VOCABULARY,
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
