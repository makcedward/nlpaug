import string

from nlpaug.augmenter.char import CharAugmenter
from nlpaug.util import Action


class RandomCharAug(CharAugmenter):
    def __init__(self, action=Action.SUBSTITUTE, name='Random_Char_Aug', aug_min=1, aug_p=0.3, tokenizer=None,
                 include_upper_case=True, include_lower_case=True, include_numeric=True, spec_char='!@#$%^&*()_+'):
        super(RandomCharAug, self).__init__(
            action=action, name=name, aug_p=aug_p, aug_min=aug_min, tokenizer=tokenizer)

        self.include_upper_case = include_upper_case
        self.include_lower_case = include_lower_case
        self.include_numeric = include_numeric
        self.spec_char = spec_char

        self.model = self.get_model()

    def insert(self, tokens):
        results = []
        for token in tokens:
            result = ''
            chars = self.tokenizer(token)

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

        # print('augment--------- start')
        # print('tokens:', tokens)
        # print('results:', results)
        # print('augment--------- end')

        return results

    def substitute(self, tokens):
        results = []
        for token in tokens:
            result = ''
            chars = self.tokenizer(token)

            if len(chars) < self.min_char:
                results.append(token)
                continue

            aug_cnt = self.generate_aug_cnt(len(chars))
            char_idxes = [i for i, char in enumerate(chars)]
            aug_idxes = self.sample(char_idxes, aug_cnt)

            result = ''.join([self.sample(self.model, 1)[0] if i in aug_idxes else char for i, char in enumerate(chars)])

            results.append(result)

        return results

    def delete(self, tokens):
        results = []
        for token in tokens:
            chars = self.tokenizer(token)

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

        return results

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
