from nlpaug.augmenter.word import WordAugmenter
from nlpaug.util import Action


class RandomWordAug(WordAugmenter):
    def __init__(self, name='Random_Word_Aug', aug_min=1, aug_p=0.3, tokenizer=None):
        super(RandomWordAug, self).__init__(
            action=Action.DELETE, name=name, aug_p=aug_p, aug_min=aug_min, tokenizer=tokenizer)

    def delete(self, tokens):
        """
        :param tokens: list of token
        :return: list of token
        """
        results = tokens.copy()

        aug_cnt = self.generate_aug_cnt(len(tokens))
        word_idxes = [i for i, t in enumerate(tokens)]
        aug_idexes = self.sample(word_idxes, aug_cnt)
        aug_idexes.sort(reverse=True)

        for aug_idx in aug_idexes:
            del results[aug_idx]

        results[0] = self.align_capitalization(tokens[0], results[0])

        return results
