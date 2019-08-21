"""
    Augmenter that apply typo error simulation to textual input.
"""

from nlpaug.augmenter.char import CharAugmenter
from nlpaug.util import Action, Method
import nlpaug.model.char as nmc


class KeyboardAug(CharAugmenter):
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
    >>> aug = nac.KeyboardAug()
    """

    def __init__(self, name='Keyboard_Aug', aug_min=1, aug_char_p=0.3, aug_word_p=0.3, stopwords=None,
                 tokenizer=None, reverse_tokenizer=None, verbose=0):
        super().__init__(
            action=Action.SUBSTITUTE, name=name, aug_char_p=aug_char_p, aug_word_p=aug_word_p, aug_min=aug_min,
            tokenizer=tokenizer, reverse_tokenizer=reverse_tokenizer, stopwords=stopwords, verbose=verbose)

        # TODO: support other type of keyboard
        self.keyboard_type = 'qwerty'
        self.model = self.get_model()

    def skip_aug(self, token_idxes, tokens):
        results = []
        for token_idx in token_idxes:
            # Some word does not come with vector. It will be excluded in lucky draw.
            char = tokens[token_idx]
            if char in self.model.model and len(self.model.predict(char)) > 1:
                results.append(token_idx)

        return results

    def substitute(self, data):
        results = []
        tokens = self.tokenizer(data)
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

                result += self.sample(self.model.predict(chars[char_i]), 1)[0]

            results.append(result)

        return self.reverse_tokenizer(results)

    @classmethod
    def get_model(cls, include_spec=True, case_sensitive=True):
        return nmc.Keyboard(include_spec=include_spec, case_sensitive=case_sensitive)