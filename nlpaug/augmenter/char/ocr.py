from nlpaug.augmenter.char import CharAugmenter
from nlpaug.util import Action, Method


class OcrAug(CharAugmenter):
    def __init__(self, name='OCR_Aug', aug_min=1, aug_char_p=0.3, aug_word_p=0.3, stopwords=[], verbose=0):
        """
        Simulate OCR error on input text.

        For example, OCR may recognize I as 1 incorrectly. Pre-defined possible OCR mapping is leveraged.

        :param name: Name of this augmenter.
        :param aug_min: Minimum number of character will be augmented.
        :param aug_char_p: Percentage of character (per token) will be augmented.
        :param aug_word_p: Percentage of word will be augmented.
        :param stopwords: List of words which will be skipped from augment operation.
        :param verbose: Verbosity mode.
        """
        super(OcrAug, self).__init__(
            action=Action.SUBSTITUTE, name=name, aug_char_p=aug_char_p, aug_word_p=aug_word_p, aug_min=aug_min,
            tokenizer=None, stopwords=stopwords, verbose=verbose)

        self.model = self.get_model()

    def skip_aug(self, token_idxes, tokens):
        results = []
        for token_idx in token_idxes:
            """
                Some character mapping do not exist. It will be excluded in lucky draw. 
            """
            char = tokens[token_idx]
            if char in self.model and len(self.model[char]) > 0:
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

