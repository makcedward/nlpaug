"""
    Augmenter that apply random word operation to textual input.
"""

from nlpaug.augmenter.word import WordAugmenter
from nlpaug.util import Action


class RandomWordAug(WordAugmenter):
    """
    Augmenter that apply randomly behavior for augmentation.

    :param str action: 'substitute', 'swap' or 'delete'. If value is 'swap', adjacent words will be swapped randomly.
        If value is 'delete', word will be removed randomly.
    :param float aug_p: Percentage of word will be augmented.
    :param int aug_min: Minimum number of word will be augmented.
    :param int aug_max: Maximum number of word will be augmented. If None is passed, number of augmentation is
        calculated via aup_p. If calculated result from aug_p is smaller than aug_max, will use calculated result from
        aug_p. Otherwise, using aug_max.
    :param list stopwords: List of words which will be skipped from augment operation.
    :param str stopwords_regex: Regular expression for matching words which will be skipped from augment operation.
    :param list target_words: List of word for replacement (used for substitute operation only). Default value is _.
    :param func tokenizer: Customize tokenization process
    :param func reverse_tokenizer: Customize reverse of tokenization process
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.word as naw
    >>> aug = naw.RandomWordAug()
    """

    def __init__(self, action=Action.DELETE, name='RandomWord_Aug', aug_min=1, aug_max=10, aug_p=0.3, stopwords=None,
                 target_words=None, tokenizer=None, reverse_tokenizer=None, stopwords_regex=None, verbose=0):
        super().__init__(
            action=action, name=name, aug_p=aug_p, aug_min=aug_min, aug_max=aug_max, stopwords=stopwords,
            tokenizer=tokenizer, reverse_tokenizer=reverse_tokenizer, device='cpu', verbose=verbose,
            stopwords_regex=stopwords_regex)

        self.target_words = ['_'] if target_words is None else target_words

    # https://arxiv.org/pdf/1711.02173.pdf, https://arxiv.org/pdf/1809.02079.pdf
    def swap(self, data):
        tokens = self.tokenizer(data)
        aug_idxes = self._get_aug_idxes(tokens)

        # https://github.com/makcedward/nlpaug/issues/76
        if len(tokens) < 2:
            return data

        for i in aug_idxes:
            original_tokens = tokens.copy()

            swap_pos = self._get_swap_position(i, len(original_tokens) - 1)
            tokens = self.change_case(tokens, i, swap_pos)

        return self.reverse_tokenizer(tokens)

    def _get_swap_position(self, pos, token_length):
        if pos == 0:
            # Force swap with next character if it is first character
            return pos + 1
        elif pos == token_length:
            # Force swap with previous character if it is last character
            return pos - 1
        else:
            return pos + self.sample([-1, 1], 1)[0]

    # https://arxiv.org/pdf/1703.02573.pdf, https://arxiv.org/pdf/1712.06751.pdf, https://arxiv.org/pdf/1806.09030.pdf
    # https://arxiv.org/pdf/1905.11268.pdf,
    def substitute(self, data):
        tokens = self.tokenizer(data)
        results = tokens.copy()

        aug_idxes = self._get_random_aug_idxes(tokens)
        aug_idxes.sort(reverse=True)

        for aug_idx in aug_idxes:
            original_token = results[aug_idx]
            results[aug_idx] = self.sample(self.target_words, 1)[0]
            if aug_idx == 0:
                results[0] = self.align_capitalization(original_token, results[0])

        return self.reverse_tokenizer(results)

    # https://arxiv.org/pdf/1905.11268.pdf, https://arxiv.org/pdf/1809.02079.pdf
    def delete(self, data):
        tokens = self.tokenizer(data)
        results = tokens.copy()

        aug_idxes = self._get_random_aug_idxes(tokens)
        aug_idxes.sort(reverse=True)

        # https://github.com/makcedward/nlpaug/issues/76
        if len(tokens) < 2:
            return data

        for aug_idx in aug_idxes:
            original_token = results[aug_idx]
            del results[aug_idx]
            if aug_idx == 0:
                results[0] = self.align_capitalization(original_token, results[0])

        return self.reverse_tokenizer(results)
