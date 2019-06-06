from nlpaug.augmenter.char import CharAugmenter
from nlpaug.util import Action


class OcrAug(CharAugmenter):
    def __init__(self, name='OCR_Aug', aug_min=1, aug_p=0.3, stopwords=[], verbose=0):
        super(OcrAug, self).__init__(
            action=Action.SUBSTITUTE, name=name, aug_p=aug_p, aug_min=aug_min, tokenizer=None, stopwords=stopwords,
            verbose=verbose)

        self.model = self.get_model()

    def substitute(self, text):
        results = []
        for token in self.tokenizer(text):
            result = ''
            chars = self.token2char(token)
            aug_cnt = self.generate_aug_cnt(len(chars))
            char_idxes = [i for i, char in enumerate(chars)]
            aug_idexes = self.sample(char_idxes, aug_cnt)

            for i, char in enumerate(chars):
                # Skip if no augment for char
                if i not in aug_idexes:
                    result += char
                    continue

                # Skip if no mapping
                if char not in self.model or len(self.model[char]) < 1:
                    result += char
                    continue

                result += self.sample(self.model[char], 1)[0]

            results.append(result)

        return self.reverse_tokenizer(results)

    def get_model(self):
        mapping = {
            '0': ['8', '9', 'o', 'O', 'D'],
            '1': ['4', '7', 'l', 'I'],
            '2': ['z', 'Z'],
            '5': ['8'],
            '6': ['b'],
            '8': ['s', 'S', '@', '&'],
            '9': ['g'],
            'o': ['u'],
            'r': ['k'],
            'C': ['G'],
            'O': ['D', 'U'],
            'E': ['B']
        }

        result = {}

        for k in mapping:
            result[k] = mapping[k]

        for k in mapping:
            for v in mapping[k]:
                if v not in result:
                    result[v] = []

                if k not in result[v]:
                    result[v].append(k)

        return result

