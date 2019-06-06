import string

from nlpaug.augmenter.char import CharAugmenter
from nlpaug.util import Action


class RandomCharAug(CharAugmenter):
    def __init__(self, action=Action.SUBSTITUTE, name='RandomChar_Aug', aug_min=1, aug_p=0.3,
                 include_upper_case=True, include_lower_case=True, include_numeric=True,
                 spec_char='!@#$%^&*()_+', stopwords=[], verbose=0):
        super(RandomCharAug, self).__init__(
            action=action, name=name, aug_p=aug_p, aug_min=aug_min, tokenizer=None, stopwords=stopwords,
            verbose=verbose)

        self.include_upper_case = include_upper_case
        self.include_lower_case = include_lower_case
        self.include_numeric = include_numeric
        self.spec_char = spec_char

        self.model = self.get_model()

    def insert(self, text):
        results = []
        for token in self.tokenizer(text):
            if token in self.stopwords:
                results.append(token)
                continue

            chars = self.token2char(token)

            if len(chars) < self.min_char:
                results.append(token)
                continue

            aug_cnt = self.generate_aug_cnt(len(chars))
            char_idxes = [i for i, char in enumerate(chars)]
            aug_idxes = self.sample(char_idxes, aug_cnt)
            aug_idxes.sort(reverse=True)

            for i in aug_idxes:
                chars.insert(i, self.sample(self.model, 1)[0])

            result = ''.join(chars)
            results.append(result)

        return self.reverse_tokenizer(results)

    def substitute(self, text):
        results = []
        for token in self.tokenizer(text):
            if token in self.stopwords:
                results.append(token)
                continue

            chars = self.token2char(token)
            if len(chars) < self.min_char:
                results.append(token)
                continue

            aug_cnt = self.generate_aug_cnt(len(chars))
            char_idxes = [i for i, char in enumerate(chars)]
            aug_idxes = self.sample(char_idxes, aug_cnt)

            result = ''.join([self.sample(self.model, 1)[0] if i in aug_idxes else char for i, char in enumerate(chars)])
            results.append(result)

        return self.reverse_tokenizer(results)

    def swap(self, text):
        results = []
        for token in self.tokenizer(text):
            if token in self.stopwords:
                results.append(token)
                continue

            chars = self.token2char(token)
            original_chars = chars.copy()

            if len(chars) < self.min_char:
                results.append(token)
                continue

            aug_cnt = self.generate_aug_cnt(len(chars))
            char_idxes = [i for i, char in enumerate(chars)]
            aug_idxes = self.sample(char_idxes, aug_cnt)

            for i in aug_idxes:
                swap_position = self._get_swap_position(i, len(chars)-1)
                is_original_upper, is_swap_upper = chars[i].isupper(), chars[swap_position].isupper()
                chars[i], chars[swap_position] = original_chars[swap_position], original_chars[i]

                # Swap case
                if is_original_upper:
                    chars[i] = chars[i].upper()
                else:
                    chars[i] = chars[i].lower()
                if is_swap_upper:
                    chars[swap_position] = chars[swap_position].upper()
                else:
                    chars[swap_position] = chars[swap_position].lower()

            result = ''.join(chars)
            results.append(result)

        return self.reverse_tokenizer(results)

    def delete(self, text):
        results = []
        for token in self.tokenizer(text):
            if token in self.stopwords:
                results.append(token)
                continue

            chars = self.token2char(token)

            if len(chars) < self.min_char:
                results.append(token)
                continue

            aug_cnt = self.generate_aug_cnt(len(chars))
            char_idxes = [i for i, char in enumerate(chars)]
            aug_idxes = self.sample(char_idxes, aug_cnt)
            aug_idxes.sort(reverse=True)

            for i in aug_idxes:
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

