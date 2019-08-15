"""
    Augmenter that apply random character error to textual input.
"""

import string

from nlpaug.augmenter.char import CharAugmenter
from nlpaug.util import Action, Method


class RandomCharAug(CharAugmenter):
    """
    Augmenter that generate character error by random values. For example, people may type i as o incorrectly.

    :param str action: Possible values are 'insert', 'substitute', 'swap' and 'delete'. If value is 'insert', a new
        character will be injected to randomly. If value is 'substitute', a random character will be replaced
        original character randomly. If value is 'swap', adjacent characters within sample word will be swapped
        randomly. If value is 'delete', character will be removed randomly.
    :param int aug_min: Minimum number of character will be augmented.
    :param float aug_char_p: Percentage of character (per token) will be augmented.
    :param float aug_word_p: Percentage of word will be augmented.
    :param bool include_upper_case: If True, upper case character may be included in augmented data.
    :param bool include_lower_case: If True, lower case character may be included in augmented data.
    :param bool include_numeric: If True, numeric character may be included in augmented data.
    :param str spec_char: Special character may be included in augmented data.
    :param list stopwords: List of words which will be skipped from augment operation.
    :param func tokenizer: Customize tokenization process
    :param func reverse_tokenizer: Customize reverse of tokenization process
    :param str name: Name of this augmenter.

    >>> import nlpaug.augmenter.char as nac
    >>> aug = nac.QwertyAug()
    """

    def __init__(self, action=Action.SUBSTITUTE, name='RandomChar_Aug', aug_min=1, aug_char_p=0.3, aug_word_p=0.3,
                 include_upper_case=True, include_lower_case=True, include_numeric=True,
                 spec_char='!@#$%^&*()_+', stopwords=[], tokenizer=None, reverse_tokenizer=None, verbose=0):
        super().__init__(
            action=action, name=name, aug_char_p=aug_char_p, aug_word_p=aug_word_p, aug_min=aug_min,
            tokenizer=tokenizer, reverse_tokenizer=reverse_tokenizer, stopwords=stopwords, verbose=verbose)

        self.include_upper_case = include_upper_case
        self.include_lower_case = include_lower_case
        self.include_numeric = include_numeric
        self.spec_char = spec_char

        self.model = self.get_model()

    def insert(self, text):
        results = []
        tokens = self.tokenizer(text)
        aug_word_idxes = self._get_aug_idxes(tokens, self.aug_word_p, Method.WORD)

        for token_i, token in enumerate(tokens):
            if token_i not in aug_word_idxes:
                results.append(token)
                continue

            chars = self.token2char(token)
            aug_char_idxes = self._get_aug_idxes(chars, self.aug_char_p, Method.CHAR)
            if aug_char_idxes is None:
                results.append(token)
                continue

            aug_char_idxes.sort(reverse=True)
            for char_i in aug_char_idxes:
                chars.insert(char_i, self.sample(self.model, 1)[0])

            result = ''.join(chars)
            results.append(result)

        return self.reverse_tokenizer(results)

    def substitute(self, text):
        results = []
        tokens = self.tokenizer(text)
        aug_word_idxes = self._get_aug_idxes(tokens, self.aug_word_p, Method.WORD)

        for token_i, token in enumerate(tokens):
            if token_i not in aug_word_idxes:
                results.append(token)
                continue

            result = ''
            chars = self.token2char(token)
            aug_char_idxes = self._get_aug_idxes(chars, self.aug_char_p, Method.CHAR)
            if aug_char_idxes is None:
                results.append(token)
                continue

            for char_i, char in enumerate(chars):
                if char_i not in aug_char_idxes:
                    result += char
                    continue

                result += self.sample(self.model, 1)[0]

            results.append(result)

        return self.reverse_tokenizer(results)

    def swap(self, text):
        results = []
        tokens = self.tokenizer(text)
        aug_word_idxes = self._get_aug_idxes(tokens, self.aug_word_p, Method.WORD)

        for token_i, token in enumerate(tokens):
            if token_i not in aug_word_idxes:
                results.append(token)
                continue

            result = ''
            chars = self.token2char(token)
            original_chars = chars.copy()

            aug_char_idxes = self._get_aug_idxes(chars, self.aug_char_p, Method.CHAR)
            if aug_char_idxes is None:
                results.append(token)
                continue

            for char_i in aug_char_idxes:
                swap_position = self._get_swap_position(char_i, len(chars)-1)
                is_original_upper, is_swap_upper = chars[char_i].isupper(), chars[swap_position].isupper()
                chars[char_i], chars[swap_position] = original_chars[swap_position], original_chars[char_i]

                # Swap case
                if is_original_upper:
                    chars[char_i] = chars[char_i].upper()
                else:
                    chars[char_i] = chars[char_i].lower()
                if is_swap_upper:
                    chars[swap_position] = chars[swap_position].upper()
                else:
                    chars[swap_position] = chars[swap_position].lower()

                result += self.sample(self.model, 1)[0]

            result = ''.join(chars)
            results.append(result)

        return self.reverse_tokenizer(results)

    def delete(self, text):
        results = []
        tokens = self.tokenizer(text)
        aug_word_idxes = self._get_aug_idxes(tokens, self.aug_word_p, Method.WORD)

        for token_i, token in enumerate(tokens):
            if token_i not in aug_word_idxes:
                results.append(token)
                continue

            chars = self.token2char(token)
            aug_char_idxes = self._get_aug_idxes(chars, self.aug_char_p, Method.CHAR)
            if aug_char_idxes is None:
                results.append(token)
                continue

            aug_char_idxes.sort(reverse=True)
            for i in aug_char_idxes:
                del chars[i]

            result = ''.join(chars)
            results.append(result)

        return self.reverse_tokenizer(results)

    def get_model(self):
        candidates = []
        if self.include_upper_case:
            candidates += string.ascii_uppercase
        if self.include_lower_case:
            candidates += string.ascii_lowercase
        if self.include_numeric:
            candidates += string.digits
        candidates += self.spec_char

        return candidates

    def _get_swap_position(self, pos, token_length):
        if pos == 0:
            # Force swap with next character if it is first character
            return pos + 1
        elif pos == token_length:
            # Force swap with previous character if it is last character
            return pos - 1
        else:
            return pos + self.sample([-1, 1], 1)[0]

