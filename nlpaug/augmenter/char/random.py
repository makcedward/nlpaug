# Source: https://arxiv.org/pdf/1711.02173.pdf

"""
    Augmenter that apply random character error to textual input.
"""

import string

from nlpaug.augmenter.char import CharAugmenter
from nlpaug.util import Action, Method, Doc


class RandomCharAug(CharAugmenter):
    # https://arxiv.org/pdf/1711.02173.pdf
    """
    Augmenter that generate character error by random values. For example, people may type i as o incorrectly.

    :param str action: Possible values are 'insert', 'substitute', 'swap' and 'delete'. If value is 'insert', a new
        character will be injected to randomly. If value is 'substitute', a random character will be replaced
        original character randomly. If value is 'swap', adjacent characters within sample word will be swapped
        randomly. If value is 'delete', character will be removed randomly.
    :param float aug_char_p: Percentage of character (per token) will be augmented.
    :param int aug_char_min: Minimum number of character will be augmented.
    :param int aug_char_max: Maximum number of character will be augmented. If None is passed, number of augmentation is
        calculated via aup_char_p. If calculated result from aug_char_p is smaller than aug_char_max, will use calculated result
        from aup_char_p. Otherwise, using aug_max.
    :param float aug_word_p: Percentage of word will be augmented.
    :param int aug_word_min: Minimum number of word will be augmented.
    :param int aug_word_max: Maximum number of word will be augmented. If None is passed, number of augmentation is
        calculated via aup_word_p. If calculated result from aug_word_p is smaller than aug_word_max, will use calculated result
        from aug_word_p. Otherwise, using aug_max.
    :param bool include_upper_case: If True, upper case character may be included in augmented data. If `candidiates'
        value is provided, this param will be ignored.
    :param bool include_lower_case: If True, lower case character may be included in augmented data. If `candidiates'
        value is provided, this param will be ignored.
    :param bool include_numeric: If True, numeric character may be included in augmented data. If `candidiates'
        value is provided, this param will be ignored.
    :param int min_char: If word less than this value, do not draw word for augmentation
    :param swap_mode: When action is 'swap', you may pass 'adjacent', 'middle' or 'random'. 'adjacent' means swap action
        only consider adjacent character (within same word). 'middle' means swap action consider adjacent character but
        not the first and last character of word. 'random' means swap action will be executed without constraint.
    :param str spec_char: Special character may be included in augmented data. If `candidiates'
        value is provided, this param will be ignored.
    :param list stopwords: List of words which will be skipped from augment operation.
    :param str stopwords_regex: Regular expression for matching words which will be skipped from augment operation.
    :param func tokenizer: Customize tokenization process
    :param func reverse_tokenizer: Customize reverse of tokenization process
    :param List candidiates: List of string for augmentation. E.g. ['AAA', '11', '===']. If values is provided,
        `include_upper_case`, `include_lower_case`, `include_numeric` and `spec_char` will be ignored.
    :param str name: Name of this augmenter.

    >>> import nlpaug.augmenter.char as nac
    >>> aug = nac.RandomCharAug()
    """

    def __init__(self, action=Action.SUBSTITUTE, name='RandomChar_Aug', aug_char_min=1, aug_char_max=10, aug_char_p=0.3,
                 aug_word_p=0.3, aug_word_min=1, aug_word_max=10, include_upper_case=True, include_lower_case=True,
                 include_numeric=True, min_char=4, swap_mode='adjacent', spec_char='!@#$%^&*()_+', stopwords=None,
                 tokenizer=None, reverse_tokenizer=None, verbose=0, stopwords_regex=None, candidiates=None):
        super().__init__(
            action=action, name=name, min_char=min_char, aug_char_min=aug_char_min, aug_char_max=aug_char_max,
            aug_char_p=aug_char_p, aug_word_min=aug_word_min, aug_word_max=aug_word_max, aug_word_p=aug_word_p,
            tokenizer=tokenizer, reverse_tokenizer=reverse_tokenizer, stopwords=stopwords, device='cpu',
            verbose=verbose, stopwords_regex=stopwords_regex, include_special_char=True, include_detail=False)

        self.include_upper_case = include_upper_case
        self.include_lower_case = include_lower_case
        self.include_numeric = include_numeric
        self.swap_mode = swap_mode
        self.spec_char = spec_char
        self.candidiates = candidiates

        self.model = self.get_model()

    def insert(self, data):
        if not data or not data.strip():
            return data

        change_seq = 0

        doc = Doc(data, self.tokenizer(data))
        aug_word_idxes = self._get_aug_idxes(
            doc.get_original_tokens(), self.aug_word_min, self.aug_word_max, self.aug_word_p, Method.WORD)

        if aug_word_idxes is None:
            return data

        for token_i, token in enumerate(doc.get_original_tokens()):
            if token_i not in aug_word_idxes:
                continue

            chars = self.token2char(token)
            aug_char_idxes = self._get_aug_idxes(chars, self.aug_char_min, self.aug_char_max, self.aug_char_p,
                                                 Method.CHAR)
            if aug_char_idxes is None:
                continue

            aug_char_idxes.sort(reverse=True)
            for char_i in aug_char_idxes:
                chars.insert(char_i, self.sample(self.model, 1)[0])

            # No capitalization alignment as this augmenter try to simulate random error

            new_token = ''.join(chars)
            change_seq += 1
            doc.add_change_log(token_i, new_token=new_token, action=Action.INSERT,
                                  change_seq=self.parent_change_seq + change_seq)

        if self.include_detail:
            return self.reverse_tokenizer(doc.get_augmented_tokens()), doc.get_change_logs()
        else:
            return self.reverse_tokenizer(doc.get_augmented_tokens())

    def substitute(self, data):
        if not data or not data.strip():
            return data

        change_seq = 0

        doc = Doc(data, self.tokenizer(data))
        aug_word_idxes = self._get_aug_idxes(
            doc.get_original_tokens(), self.aug_word_min, self.aug_word_max, self.aug_word_p, Method.WORD)

        if aug_word_idxes is None:
            return data

        for token_i, token in enumerate(doc.get_original_tokens()):
            if token_i not in aug_word_idxes:
                continue

            substitute_token = ''
            chars = self.token2char(token)
            aug_char_idxes = self._get_aug_idxes(chars, self.aug_char_min, self.aug_char_max, self.aug_char_p,
                                                 Method.CHAR)
            if aug_char_idxes is None:
                continue

            for char_i, char in enumerate(chars):
                if char_i not in aug_char_idxes:
                    substitute_token += char
                    continue

                substitute_token += self.sample(self.model, 1)[0]

            # No capitalization alignment as this augmenter try to simulate random error

            change_seq += 1
            doc.add_change_log(token_i, new_token=substitute_token, action=Action.SUBSTITUTE,
                               change_seq=self.parent_change_seq + change_seq)

        if self.include_detail:
            return self.reverse_tokenizer(doc.get_augmented_tokens()), doc.get_change_logs()
        else:
            return self.reverse_tokenizer(doc.get_augmented_tokens())

    def swap(self, data):
        if not data or not data.strip():
            return data

        change_seq = 0

        doc = Doc(data, self.tokenizer(data))
        aug_word_idxes = self._get_aug_idxes(
            doc.get_original_tokens(), self.aug_word_min, self.aug_word_max, self.aug_word_p, Method.WORD)

        if aug_word_idxes is None:
            return data

        for token_i, token in enumerate(doc.get_original_tokens()):
            if token_i not in aug_word_idxes:
                continue

            swap_token = ''
            chars = self.token2char(token)

            aug_char_idxes = self._get_aug_idxes(chars, self.aug_char_min, self.aug_char_max, self.aug_char_p,
                                                 Method.CHAR)
            if aug_char_idxes is None or len(aug_char_idxes) < 1:
                continue

            for char_i in aug_char_idxes:
                swap_position = self._get_swap_position(char_i, len(chars)-1, mode=self.swap_mode)
                is_original_upper, is_swap_upper = chars[char_i].isupper(), chars[swap_position].isupper()
                original_chars = chars.copy()
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

                swap_token += self.sample(self.model, 1)[0]

            # No capitalization alignment as this augmenter try to simulate random error

            swap_token = ''.join(chars)
            change_seq += 1
            doc.add_change_log(token_i, new_token=swap_token, action=Action.SWAP,
                               change_seq=self.parent_change_seq + change_seq)

        if self.include_detail:
            return self.reverse_tokenizer(doc.get_augmented_tokens()), doc.get_change_logs()
        else:
            return self.reverse_tokenizer(doc.get_augmented_tokens())

    def delete(self, data):
        if not data or not data.strip():
            return data
            
        change_seq = 0

        doc = Doc(data, self.tokenizer(data))
        aug_word_idxes = self._get_aug_idxes(
            doc.get_original_tokens(), self.aug_word_min, self.aug_word_max, self.aug_word_p, Method.WORD)

        if aug_word_idxes is None:
            return data

        for token_i, token in enumerate(doc.get_original_tokens()):
            if token_i not in aug_word_idxes:
                continue

            chars = self.token2char(token)
            aug_char_idxes = self._get_aug_idxes(chars, self.aug_char_min, self.aug_char_max, self.aug_char_p,
                                                 Method.CHAR)
            if aug_char_idxes is None or len(aug_char_idxes) < 1:
                continue

            aug_char_idxes.sort(reverse=True)
            for i in aug_char_idxes:
                del chars[i]

            # No capitalization alignment as this augmenter try to simulate random error

            delete_token = ''.join(chars)
            change_seq += 1
            doc.add_change_log(token_i, new_token=delete_token, action=Action.DELETE,
                               change_seq=self.parent_change_seq + change_seq)

        if self.include_detail:
            return self.reverse_tokenizer(doc.get_augmented_tokens()), doc.get_change_logs()
        else:
            return self.reverse_tokenizer(doc.get_augmented_tokens())

    def get_model(self):
        if self.candidiates:
            return self.candidiates

        candidates = []
        if self.include_upper_case:
            candidates += string.ascii_uppercase
        if self.include_lower_case:
            candidates += string.ascii_lowercase
        if self.include_numeric:
            candidates += string.digits
        if self.spec_char:
            candidates += self.spec_char

        return candidates

    def _get_swap_position(self, pos, token_length, mode='adjacent'):
        if mode == 'adjacent':
            if pos == 0:
                # Force swap with next character if it is first character
                return pos + 1
            elif pos == token_length:
                # Force swap with previous character if it is last character
                return pos - 1
            else:
                return pos + self.sample([-1, 1], 1)[0]
        elif mode == 'middle':
            # Middle Random: https://arxiv.org/pdf/1711.02173.pdf
            candidates = [_ for _ in range(token_length) if _ not in [0, pos, token_length]]
            if len(candidates) == 0:
                return pos
            return self.sample(candidates, 1)[0]
        elif mode == 'random':
            # Fully Random: https://arxiv.org/pdf/1711.02173.pdf
            candidates = [_ for _ in range(token_length) if _ not in [pos]]
            if len(candidates) < 1:
                return pos
            return self.sample(candidates, 1)[0]
