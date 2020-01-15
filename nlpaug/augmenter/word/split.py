# Source: https://arxiv.org/pdf/1812.05271v1.pdf

"""
    Augmenter that apply word splitting operation to textual input.
"""

from nlpaug.augmenter.word import WordAugmenter
from nlpaug.util import Action


class SplitAug(WordAugmenter):
    """
    Augmenter that apply word splitting for augmentation.

    :param float aug_p: Percentage of word will be augmented.
    :param int aug_min: Minimum number of word will be augmented.
    :param int aug_max: Maximum number of word will be augmented. If None is passed, number of augmentation is
        calculated via aup_p. If calculated result from aug_p is smaller than aug_max, will use calculated result from
        aug_p. Otherwise, using aug_max.
    :param int min_char: If word less than this value, do not draw word for augmentation
    :param list stopwords: List of words which will be skipped from augment operation.
    :param str stopwords_regex: Regular expression for matching words which will be skipped from augment operation.
    :param func tokenizer: Customize tokenization process
    :param func reverse_tokenizer: Customize reverse of tokenization process
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.word as naw
    >>> aug = naw.SplitAug()
    """

    def __init__(self, name='Split_Aug', aug_min=1, aug_max=10, aug_p=0.3, min_char=4, stopwords=None,
                 tokenizer=None, reverse_tokenizer=None, stopwords_regex=None, verbose=0):
        super().__init__(
            action=Action.SPLIT, name=name, aug_p=aug_p, aug_min=aug_min, aug_max=aug_max, stopwords=stopwords,
            tokenizer=tokenizer, reverse_tokenizer=reverse_tokenizer, device='cpu', verbose=verbose,
            stopwords_regex=stopwords_regex)

        self.min_char = min_char

    def skip_aug(self, token_idxes, tokens):
        results = []
        for token_idx in token_idxes:
            if len(tokens[token_idx]) >= self.min_char:
                results.append(token_idx)
        return results

    def split(self, data):
        tokens = self.tokenizer(data)
        results = tokens.copy()

        aug_idxes = self._get_aug_idxes(tokens)
        aug_idxes.sort(reverse=True)

        for aug_idx in aug_idxes:
            separate_pos = self.sample(len(tokens[aug_idx]), 1)
            prev_token = tokens[aug_idx][:separate_pos]
            next_token = tokens[aug_idx][separate_pos:]

            results[aug_idx] = next_token
            results.insert(aug_idx, prev_token)

        return self.reverse_tokenizer(results)
