"""
    Augmenter that apply random word operation to textual input.
"""

from nlpaug.augmenter.word import WordAugmenter
from nlpaug.util import Action


class RandomWordAug(WordAugmenter):
    """
    Augmenter that apply randomly behavior for augmentation.

    :param str action: Either 'swap' or 'delete'. If value is 'swap', adjacent words will be swapped randomly.
        If value is 'delete', word will be removed randomly.
    :param int aug_min: Minimum number of word will be augmented.
    :param float aug_p: Percentage of word will be augmented.
    :param list stopwords: List of words which will be skipped from augment operation.
    :param func tokenizer: Customize tokenization process
    :param func reverse_tokenizer: Customize reverse of tokenization process
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.word as naw
    >>> aug = naw.RandomWordAug()
    """

    def __init__(self, action=Action.DELETE, name='RandomWord_Aug', aug_min=1, aug_p=0.3, stopwords=None,
                 tokenizer=None, reverse_tokenizer=None, verbose=0):
        super().__init__(
            action=action, name=name, aug_p=aug_p, aug_min=aug_min, stopwords=stopwords,
            tokenizer=tokenizer, reverse_tokenizer=reverse_tokenizer, verbose=verbose)

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
