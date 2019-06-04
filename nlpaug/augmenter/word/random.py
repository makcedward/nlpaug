from nlpaug.augmenter.word import WordAugmenter
from nlpaug.util import Action


class RandomWordAug(WordAugmenter):
    def __init__(self, name='RandomWord_Aug', aug_min=1, aug_p=0.3, tokenizer=None, stopwords=[]):
        super(RandomWordAug, self).__init__(
            action=Action.DELETE, name=name, aug_p=aug_p, aug_min=aug_min, tokenizer=tokenizer, stopwords=stopwords)

    def delete(self, text):
        """
        :param text: input
        :return: list of token
        """
        tokens = self.tokenizer(text)
        results = tokens.copy()

        aug_idexes = self._get_aug_idxes(tokens)
        aug_idexes.sort(reverse=True)

        for aug_idx in aug_idexes:
            del results[aug_idx]

        results[0] = self.align_capitalization(tokens[0], results[0])

        return self.reverse_tokenizer(results)
