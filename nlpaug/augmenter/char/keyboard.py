"""
    Augmenter that apply typo error simulation to textual input.
"""

from nlpaug.augmenter.char import CharAugmenter
from nlpaug.util import Action, Method
import nlpaug.model.char as nmc


class KeyboardAug(CharAugmenter):
    # https://arxiv.org/pdf/1711.02173.pdf
    """
    Augmenter that simulate typo error by random values. For example, people may type i as o incorrectly.\
        One keyboard distance is leveraged to replace character by possible keyboard error.

    :param float aug_char_p: Percentage of character (per token) will be augmented.
    :param int aug_char_min: Minimum number of character will be augmented.
    :param int aug_char_max: Maximum number of character will be augmented. If None is passed, number of augmentation is
        calculated via aup_char_p. If calculated result from aug_p is smaller than aug_max, will use calculated result
        from aup_char_p. Otherwise, using aug_max.
    :param float aug_word_p: Percentage of word will be augmented.
    :param int aug_word_min: Minimum number of word will be augmented.
    :param int aug_word_max: Maximum number of word will be augmented. If None is passed, number of augmentation is
        calculated via aup_word_p. If calculated result from aug_p is smaller than aug_max, will use calculated result
        from aug_word_p. Otherwise, using aug_max.
    :param list stopwords: List of words which will be skipped from augment operation.
    :param str stopwords_regex: Regular expression for matching words which will be skipped from augment operation.
    :param func tokenizer: Customize tokenization process
    :param func reverse_tokenizer: Customize reverse of tokenization process
    :param bool special_char: Include special character
    :param bool numeric: Include numeric character
    :param bool upper_case: Include upper case character
    :param str model_path: Loading customize model from file system
    :param str lang: Indicate built-in language model. If custom model is used (passing model_path), this value will
        be ignored.
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.char as nac
    >>> aug = nac.KeyboardAug()
    """

    def __init__(self, name='Keyboard_Aug', aug_char_min=1, aug_char_max=10, aug_char_p=0.3,
                 aug_word_p=0.3, aug_word_min=1, aug_word_max=10, stopwords=None,
                 tokenizer=None, reverse_tokenizer=None, special_char=True, numeric=True,
                 upper_case=True, lang="en", verbose=0, stopwords_regex=None, model_path=None):
        super().__init__(
            action=Action.SUBSTITUTE, name=name, aug_char_min=aug_char_min, aug_char_max=aug_char_max,
            aug_char_p=aug_char_p, aug_word_min=aug_word_min, aug_word_max=aug_word_max, aug_word_p=aug_word_p,
            tokenizer=tokenizer, reverse_tokenizer=reverse_tokenizer, stopwords=stopwords, device='cpu',
            verbose=verbose, stopwords_regex=stopwords_regex)

        # TODO: support other type of keyboard
        self.keyboard_type = 'qwerty'
        self.special_char = special_char
        self.numeric = numeric
        self.upper_case = upper_case
        self.lang = lang
        self.model_path = model_path
        self.model = self.get_model(special_char, numeric, upper_case, lang, model_path)

    def skip_aug(self, token_idxes, tokens):
        results = []
        for token_idx in token_idxes:
            char = tokens[token_idx]
            if char in self.model.model and len(self.model.predict(char)) > 0:
                results.append(token_idx)

        return results

    def substitute(self, data):
        results = []
        tokens = self.tokenizer(data)
        aug_word_idxes = self._get_aug_idxes(
            tokens, self.aug_word_min, self.aug_word_max, self.aug_word_p, Method.WORD)

        for token_i, token in enumerate(tokens):
            if token_i not in aug_word_idxes:
                results.append(token)
                continue

            result = ''
            chars = self.token2char(token)
            aug_char_idxes = self._get_aug_idxes(chars, self.aug_char_min, self.aug_char_max, self.aug_char_p,
                                                 Method.CHAR)
            if aug_char_idxes is None:
                results.append(token)
                continue

            for char_i, char in enumerate(chars):
                if char_i not in aug_char_idxes:
                    result += char
                    continue

                result += self.sample(self.model.predict(chars[char_i]), 1)[0]

            # No capitalization alignment as this augmenter try to simulate typo

            results.append(result)

        return self.reverse_tokenizer(results)

    @classmethod
    def get_model(cls, special_char=True, numeric=True, upper_case=True, lang="en", model_path=None):
        return nmc.Keyboard(special_char=special_char, numeric=numeric, upper_case=upper_case, lang=lang,
                            model_path=model_path)
