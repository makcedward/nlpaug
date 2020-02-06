"""
    Augmenter that apply semantic meaning based to textual input.
"""

import os

from nlpaug.augmenter.word import WordAugmenter
from nlpaug.util import Action, PartOfSpeech, WarningException, WarningName, WarningCode, WarningMessage
import nlpaug.model.word_dict as nmw

PPDB_MODEL = {}


def init_ppdb_model(dict_path, force_reload=False):
    # Load model once at runtime
    global PPDB_MODEL

    model_name = os.path.basename(dict_path)
    if model_name in PPDB_MODEL and not force_reload:
        return PPDB_MODEL

    model = nmw.Ppdb(dict_path)
    PPDB_MODEL[model_name] = model

    return model


class SynonymAug(WordAugmenter):
    # https://arxiv.org/pdf/1809.02079.pdf
    """
    Augmenter that leverage semantic meaning to substitute word.

    :param str aug_src: Support 'wordnet' and 'ppdb' .
    :param str model_path: Path of dictionary. Mandatory field if using PPDB as data source
    :param str lang: Language of your text. Default value is 'eng'.
    :param float aug_p: Percentage of word will be augmented.
    :param int aug_min: Minimum number of word will be augmented.
    :param int aug_max: Maximum number of word will be augmented. If None is passed, number of augmentation is
        calculated via aup_p. If calculated result from aug_p is smaller than aug_max, will use calculated result from
        aug_p. Otherwise, using aug_max.
    :param list stopwords: List of words which will be skipped from augment operation.
    :param str stopwords_regex: Regular expression for matching words which will be skipped from augment operation.
    :param func tokenizer: Customize tokenization process
    :param func reverse_tokenizer: Customize reverse of tokenization process
    :param bool force_reload: Force reload model to memory when initialize the class.
        Default value is False and suggesting to keep it as False if performance is the consideration.
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.word as naw
    >>> aug = naw.SynonymAug()
    """

    def __init__(self, aug_src='wordnet', model_path=None, name='Synonym_Aug', aug_min=1, aug_max=10, aug_p=0.3,
                 lang='eng', stopwords=None, tokenizer=None, reverse_tokenizer=None, stopwords_regex=None,
                 force_reload=False, verbose=0):
        super().__init__(
            action=Action.SUBSTITUTE, name=name, aug_p=aug_p, aug_min=aug_min, aug_max=aug_max, stopwords=stopwords,
            tokenizer=tokenizer, reverse_tokenizer=reverse_tokenizer, device='cpu', verbose=verbose,
            stopwords_regex=stopwords_regex)

        self.aug_src = aug_src
        self.model_path = model_path
        self.lang = lang
        self.model = self.get_model(aug_src, lang, model_path, force_reload)

    def skip_aug(self, token_idxes, tokens):
        results = []
        for token_idx in token_idxes:
            # Some word does not come with synonym/ antony. It will be excluded in lucky draw.
            if tokens[token_idx][1] not in ['DT']:
                results.append(token_idx)

        return results

    def _get_aug_idxes(self, tokens):
        aug_cnt = self.generate_aug_cnt(len(tokens))
        word_idxes = self.pre_skip_aug(tokens, tuple_idx=0)
        word_idxes = self.skip_aug(word_idxes, tokens)
        if len(word_idxes) == 0:
            if self.verbose > 0:
                exception = WarningException(name=WarningName.OUT_OF_VOCABULARY,
                                             code=WarningCode.WARNING_CODE_002, msg=WarningMessage.NO_WORD)
                exception.output()
            return None
        if len(word_idxes) < aug_cnt:
            aug_cnt = len(word_idxes)
        aug_idexes = self.sample(word_idxes, aug_cnt)
        return aug_idexes

    def substitute(self, data):
        results = []

        tokens = self.tokenizer(data)
        pos = self.model.pos_tag(tokens)

        aug_idxes = self._get_aug_idxes(pos)
        if aug_idxes is None:
            return data

        for i, original_token in enumerate(tokens):
            # Skip if no augment for word
            if i not in aug_idxes:
                results.append(original_token)
                continue

            word_poses = PartOfSpeech.constituent2pos(pos[i][1])
            candidates = []
            if word_poses is None or len(word_poses) == 0:
                # Use every possible words as the mapping does not defined correctly
                candidates.extend(self.model.predict(pos[i][0]))
            else:
                for word_pos in word_poses:
                    candidates.extend(self.model.predict(pos[i][0], pos=word_pos))

            candidates = [c for c in candidates if c.lower() != original_token.lower()]

            if len(candidates) == 0:
                results.append(original_token)
            else:
                candidate = self.sample(candidates, 1)[0]
                candidate = candidate.replace("_", " ").replace("-", " ").lower()
                results.append(self.align_capitalization(original_token, candidate))

            if i == 0:
                results[0] = self.align_capitalization(original_token, results[0])

        return self.reverse_tokenizer(results)

    @classmethod
    def get_model(cls, aug_src, lang, dict_path, force_reload):
        if aug_src == 'wordnet':
            return nmw.WordNet(lang=lang, is_synonym=True)
        elif aug_src == 'ppdb':
            return init_ppdb_model(dict_path=dict_path, force_reload=force_reload)

        raise ValueError('aug_src is not one of `wordnet` or `ppdb` while {} is passed.'.format(aug_src))

    def __str__(self):
        return 'Name:{}, Aug Src:{}, Action:{}, Method:{}'.format(self.name, self.aug_src, self.action, self.method)