from nlpaug.util import Method
from nlpaug import Augmenter


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

    def align_capitalization(self, src_token, dest_token):
        """
            Simulate capitalized string
        """
        if src_token[0].isupper() and len(src_token) > 1 and src_token[1].isupper():
            return dest_token.upper()
        elif src_token[0].isupper():
            return dest_token.capitalize()
        else:
            return dest_token

    def _get_aug_idxes(self, tokens):
        aug_cnt = self.generate_aug_cnt(len(tokens))
        word_idxes = [i for i, t in enumerate(tokens) if t not in self.stopwords]
        aug_idexes = self.sample(word_idxes, aug_cnt)
        return aug_idexes
