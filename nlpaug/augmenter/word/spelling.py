"""
    Augmenter that apply spelling error simulation to textual input.
"""

import nlpaug.model.word_dict as nmwd
from nlpaug.augmenter.word import WordAugmenter
from nlpaug.util import Action

SPELLING_ERROR_MODEL = {}


def init_spelling_error_model(dict_path, include_reverse, force_reload=False):
    # Load model once at runtime
    global SPELLING_ERROR_MODEL
    if SPELLING_ERROR_MODEL and not force_reload:
        return SPELLING_ERROR_MODEL

    spelling_error_model = nmwd.Spelling(dict_path, include_reverse)

    SPELLING_ERROR_MODEL = spelling_error_model

    return SPELLING_ERROR_MODEL


class SpellingAug(WordAugmenter):
    """
    Augmenter that leverage pre-defined spelling mistake dictionary to simulate spelling mistake.

    :param str action: Either 'swap' or 'delete'. If value is 'swap', adjacent words will be swapped randomly.
        If value is 'delete', word will be removed randomly.
    :param int aug_min: Minimum number of word will be augmented.
    :param float aug_p: Percentage of word will be augmented.
    :param list stopwords: List of words which will be skipped from augment operation.
    :param func tokenizer: Customize tokenization process
    :param func reverse_tokenizer: Customize reverse of tokenization process
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.word as naw
    >>> aug = naw.SpellingAug(dict_path='./spelling_en.txt')
    """

    def __init__(self, dict_path, name='Spelling_Aug', aug_min=1, aug_p=0.3, stopwords=None,
                 tokenizer=None, reverse_tokenizer=None, include_reverse=True, verbose=0):
        super().__init__(
            action=Action.SUBSTITUTE, name=name, aug_p=aug_p, aug_min=aug_min, stopwords=stopwords,
            tokenizer=tokenizer, reverse_tokenizer=reverse_tokenizer, verbose=verbose)

        self.dict_path = dict_path
        self.include_reverse = include_reverse
        self.model = self.get_model(force_reload=False)

    def skip_aug(self, token_idxes, tokens):
        results = []
        for token_idx in token_idxes:
            """
                Some words do not exit. It will be excluded in lucky draw. 
            """
            token = tokens[token_idx]
            if token in self.model.dict and len(self.model.dict[token]) > 0:
                results.append(token_idx)

        return results

    def substitute(self, data):
        results = []

        tokens = self.tokenizer(data)
        aug_idexes = self._get_aug_idxes(tokens)

        if aug_idexes is None:
            return data

        for i, token in enumerate(tokens):
            # Skip if no augment for word
            if i not in aug_idexes:
                results.append(token)
                continue

            candidate_words = self.model.predict(token)
            results.append(self.sample(candidate_words, 1)[0])

        return self.reverse_tokenizer(results)

    def get_model(self, force_reload):
        return init_spelling_error_model(self.dict_path, self.include_reverse, force_reload)
