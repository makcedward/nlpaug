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
    :param list target_words: List of word for replacement (used for substitute operation only). Default value is _.
    :param func tokenizer: Customize tokenization process
    :param func reverse_tokenizer: Customize reverse of tokenization process
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.word as naw
    >>> aug = naw.RandomWordAug()
    """

    def __init__(self, action=Action.DELETE, name='RandomWord_Aug', aug_min=1, aug_p=0.3, stopwords=None,
                 target_words=None, tokenizer=None, reverse_tokenizer=None, verbose=0):
        super().__init__(
            action=action, name=name, aug_p=aug_p, aug_min=aug_min, stopwords=stopwords,
            tokenizer=tokenizer, reverse_tokenizer=reverse_tokenizer, device='cpu', verbose=verbose)

        self.target_words = ['_'] if target_words is None else target_words

    def swap(self, data):
        results = self.tokenizer(data)
        aug_idxes = self._get_aug_idxes(results)
        original_tokens = results.copy()

        for i in aug_idxes:
            swap_position = self._get_swap_position(i, len(original_tokens) - 1)
            if len(results[i]) > 0:
                is_original_capitalize, is_swap_capitalize = results[i][0].isupper(), results[swap_position][0].isupper()
            else:
                is_original_capitalize = False
                is_swap_capitalize = False

            is_original_upper, is_swap_upper = results[i].isupper(), results[swap_position].isupper()
            results[i], results[swap_position] = original_tokens[swap_position], original_tokens[i]

            # Swap case
            if is_original_upper:
                results[i] = results[i].upper()
            elif is_original_capitalize:
                results[i] = results[i].capitalize()
            else:
                results[i] = results[i].lower()
            if is_swap_upper:
                results[swap_position] = results[swap_position].upper()
            elif is_swap_capitalize:
                results[swap_position] = results[swap_position].capitalize()
            else:
                results[swap_position] = results[swap_position].lower()

        return self.reverse_tokenizer(results)

    def _get_swap_position(self, pos, token_length):
        if pos == 0:
            # Force swap with next character if it is first character
            return pos + 1
        elif pos == token_length:
            # Force swap with previous character if it is last character
            return pos - 1
        else:
            return pos + self.sample([-1, 1], 1)[0]

    # https://arxiv.org/pdf/1703.02573.pdf
    def substitute(self, data):
        tokens = self.tokenizer(data)
        results = tokens.copy()

        aug_idxes = self._get_random_aug_idxes(tokens)
        aug_idxes.sort(reverse=True)

        for aug_idx in aug_idxes:
            results[aug_idx] = self.sample(self.target_words, 1)[0]

        if len(results) > 0 and len(results[0]) > 0:
            results[0] = self.align_capitalization(tokens[0], results[0])

        return self.reverse_tokenizer(results)

    def delete(self, data):
        tokens = self.tokenizer(data)
        results = tokens.copy()

        aug_idxes = self._get_random_aug_idxes(tokens)
        aug_idxes.sort(reverse=True)

        for aug_idx in aug_idxes:
            del results[aug_idx]

        if len(results) > 0 and len(results[0]) > 0:
            results[0] = self.align_capitalization(tokens[0], results[0])

        return self.reverse_tokenizer(results)
