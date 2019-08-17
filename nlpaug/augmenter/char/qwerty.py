"""
    Augmenter that apply typo error simulation to textual input.
"""

import re

from nlpaug.augmenter.char import CharAugmenter
from nlpaug.util import Action, Method
from nlpaug.util.decorator.deprecation import deprecated


@deprecated(deprecate_from='0.0.7', deprecate_to='0.0.9', msg="Use KeyboardAug from 0.0.7 version")
class QwertyAug(CharAugmenter):
    """
    Augmenter that simulate typo error by random values. For example, people may type i as o incorrectly.\
        One keyboard distance is leveraged to replace character by possible keyboard error.

    :param int aug_min: Minimum number of character will be augmented.
    :param float aug_char_p: Percentage of character (per token) will be augmented.
    :param float aug_word_p: Percentage of word will be augmented.
    :param list stopwords: List of words which will be skipped from augment operation.
    :param func tokenizer: Customize tokenization process
    :param func reverse_tokenizer: Customize reverse of tokenization process
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.char as nac
    >>> aug = nac.QwertyAug()
    """

    def __init__(self, name='Qwerty_Aug', aug_min=1, aug_char_p=0.3, aug_word_p=0.3, stopwords=None,
                 tokenizer=None, reverse_tokenizer=None, verbose=0):
        super().__init__(
            action=Action.SUBSTITUTE, name=name, aug_char_p=aug_char_p, aug_word_p=aug_word_p, aug_min=aug_min,
            tokenizer=tokenizer, reverse_tokenizer=reverse_tokenizer, stopwords=stopwords, verbose=verbose)

        self.model = self.get_model()

    def skip_aug(self, token_idxes, tokens):
        results = []
        for token_idx in token_idxes:
            # Some word does not come with vector. It will be excluded in lucky draw.
            char = tokens[token_idx]
            if char in self.model and len(self.model[char]) > 1:
                results.append(token_idx)

        return results

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

                result += self.sample(self.model[chars[char_i]], 1)[0]

            results.append(result)

        return self.reverse_tokenizer(results)

    # TUNEME: pre calcualte it
    def get_model(self, include_spec=True, case_sensitive=True):
        mapping = {
            '1': ['!', '2', '@', 'q', 'w'],
            '2': ['@', '1', '!', '3', '#', 'q', 'w', 'e'],
            '3': ['#', '2', '@', '4', '$', 'w', 'e'],
            '4': ['$', '3', '#', '5', '%', 'e', 'r'],
            '5': ['%', '4', '$', '6', '^', 'r', 't', 'y'],
            '6': ['^', '5', '%', '7', '&', 't', 'y', 'u'],
            '7': ['&', '6', '^', '8', '*', 'y', 'u', 'i'],
            '8': ['*', '7', '&', '9', '(', 'u', 'i', 'o'],
            '9': ['(', '8', '*', '0', ')', 'i', 'o', 'p'],

            'q': ['1', '!', '2', '@', 'w', 'a', 's'],
            'w': ['1', '!', '2', '@', '3', '#', 'q', 'e', 'a', 's', 'd'],
            'e': ['2', '@', '3', '#', '4', '$', 'w', 'r', 's', 'd', 'f'],
            'r': ['3', '#', '4', '$', '5', '%', 'e', 't', 'd', 'f', 'g'],
            't': ['4', '$', '5', '%', '6', '^', 'r', 'y', 'f', 'g', 'h'],
            'y': ['5', '%', '6', '^', '7', '&', 't', 'u', 'g', 'h', 'j'],
            'u': ['6', '^', '7', '&', '8', '*',' t', 'i', 'h', 'j', 'k'],
            'i': ['7', '&', '8', '*', '9', '(', 'u', 'o', 'j', 'k', 'l'],
            'o': ['8', '*', '9', '(', '0', ')', 'i', 'p', 'k', 'l'],
            'p': ['9', '(', '0', ')', 'o', 'l'],

            'a': ['q', 'w', 'a', 's', 'z', 'x'],
            's': ['q', 'w', 'e', 'a', 'd', 'z', 'x', 'c'],
            'd': ['w', 'e', 'r', 's', 'f', 'x', 'c', 'v'],
            'f': ['e', 'r', 't', 'd', 'g', 'c', 'v', 'b'],
            'g': ['r', 't', 'y', 'f', 'h', 'v', 'b', 'n'],
            'h': ['t', 'y', 'u', 'g', 'j', 'b', 'n', 'm'],
            'j': ['y', 'u', 'i', 'h', 'k', 'n', 'm', ',', '<'],
            'k': ['u', 'i', 'o', 'j', 'l', 'm', ',', '<', '.', '>'],
            'l': ['i', 'o', 'p', 'k', ';', ':', ',', '<', '.', '>', '/', '?'],

            'z': ['a', 's', 'x'],
            'x': ['a', 's', 'd', 'z', 'c'],
            'c': ['s', 'd', 'f', 'x', 'v'],
            'v': ['d', 'f', 'g', 'c', 'b'],
            'b': ['f', 'g', 'h', 'v', 'n'],
            'n': ['g', 'h', 'j', 'b', 'm'],
            'm': ['h', 'j', 'k', 'n', ',', '<']
        }

        result = {}

        for k in mapping:
            result[k] = []
            for v in mapping[k]:
                if not include_spec and not re.match("^[a-zA-Z0-9]*$", v):
                    continue

                result[k].append(v)

                if case_sensitive:
                    result[k].append(v.upper())

            result[k] = list(set(result[k]))

            if case_sensitive and re.match("^[a-z]*$", k):
                result[k.upper()] = []

                for v in mapping[k]:
                    if not include_spec and not re.match("^[a-zA-Z0-9]*$", v):
                        continue

                    k = k.upper()

                    result[k].append(v)

                    if case_sensitive:
                        result[k].append(v.upper())

                result[k] = list(set(result[k]))

        return result