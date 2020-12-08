"""
    Augmenter that apply spelling error simulation to textual input.
"""

import os

import nlpaug
import nlpaug.model.word_dict as nmwd
from nlpaug.augmenter.word import WordAugmenter
from nlpaug.util import Action, Doc, LibraryUtil

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
    # https://arxiv.org/ftp/arxiv/papers/1812/1812.04718.pdf
    """
    Augmenter that leverage pre-defined spelling mistake dictionary to simulate spelling mistake.

    :param str dict_path: Path of misspelling dictionary
    :param float aug_p: Percentage of word will be augmented.
    :param int aug_min: Minimum number of word will be augmented.
    :param int aug_max: Maximum number of word will be augmented. If None is passed, number of augmentation is
        calculated via aup_p. If calculated result from aug_p is smaller than aug_max, will use calculated result from
        aug_p. Otherwise, using aug_max.
    :param list stopwords: List of words which will be skipped from augment operation.
    :param str stopwords_regex: Regular expression for matching words which will be skipped from augment operation.
    :param func tokenizer: Customize tokenization process
    :param func reverse_tokenizer: Customize reverse of tokenization process
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.word as naw
    >>> aug = naw.SpellingAug(dict_path='./spelling_en.txt')
    """

    def __init__(self, dict_path=None, name='Spelling_Aug', aug_min=1, aug_max=10, aug_p=0.3, stopwords=None,
                 tokenizer=None, reverse_tokenizer=None, include_reverse=True, stopwords_regex=None,
                 verbose=0):
        super().__init__(
            action=Action.SUBSTITUTE, name=name, aug_p=aug_p, aug_min=aug_min, aug_max=aug_max, stopwords=stopwords,
            tokenizer=tokenizer, reverse_tokenizer=reverse_tokenizer, device='cpu', verbose=verbose,
            stopwords_regex=stopwords_regex, include_detail=False)


        self.dict_path = dict_path if dict_path else os.path.join(LibraryUtil.get_res_dir(), 'word', 'spelling', 'spelling_en.txt')
        self.include_reverse = include_reverse
        self.model = self.get_model(force_reload=False)

    def skip_aug(self, token_idxes, tokens):
        results = []
        for token_idx in token_idxes:
            # Some words do not exit. It will be excluded in lucky draw.
            token = tokens[token_idx]
            if token in self.model.dict and len(self.model.dict[token]) > 0:
                results.append(token_idx)

        return results

    def substitute(self, data):
        if not data or not data.strip():
            return data
            
        change_seq = 0
        doc = Doc(data, self.tokenizer(data))

        aug_idxes = self._get_aug_idxes(doc.get_original_tokens())

        if aug_idxes is None or len(aug_idxes) == 0:
            if self.include_detail:
                return data, []
            return data

        for aug_idx, original_token in enumerate(doc.get_original_tokens()):
            # Skip if no augment for word
            if aug_idx not in aug_idxes:
                continue

            candidate_words = self.model.predict(original_token)
            substitute_token = ''
            if candidate_words:
                substitute_token = self.sample(candidate_words, 1)[0]
            else:
                # Unexpected scenario. Adding original token
                substitute_token = original_token

            if aug_idx == 0:
                substitute_token = self.align_capitalization(original_token, substitute_token)

            change_seq += 1
            doc.add_change_log(aug_idx, new_token=substitute_token, action=Action.SUBSTITUTE,
                               change_seq=self.parent_change_seq + change_seq)

        if self.include_detail:
            return self.reverse_tokenizer(doc.get_augmented_tokens()), doc.get_change_logs()
        else:
            return self.reverse_tokenizer(doc.get_augmented_tokens())

    def get_model(self, force_reload):
        return init_spelling_error_model(self.dict_path, self.include_reverse, force_reload)
