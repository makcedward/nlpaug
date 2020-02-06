"""
    Augmenter that apply semantic meaning based to textual input.
"""

from nlpaug.augmenter.word import WordAugmenter
from nlpaug.util import Action, PartOfSpeech, WarningException, WarningName, WarningCode, WarningMessage
import nlpaug.model.word_dict as nmw


class AntonymAug(WordAugmenter):
    # https://arxiv.org/pdf/1809.02079.pdf
    """
    Augmenter that leverage semantic meaning to substitute word.

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
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.word as naw
    >>> aug = naw.AntonymAug()
    """

    def __init__(self, name='Antonym_Aug', aug_min=1, aug_max=10, aug_p=0.3, lang='eng',
                 stopwords=None, tokenizer=None, reverse_tokenizer=None, stopwords_regex=None, verbose=0):
        super().__init__(
            action=Action.SUBSTITUTE, name=name, aug_p=aug_p, aug_min=aug_min, aug_max=aug_max, stopwords=stopwords,
            tokenizer=tokenizer, reverse_tokenizer=reverse_tokenizer, device='cpu', verbose=verbose,
            stopwords_regex=stopwords_regex)

        self.aug_src = 'wordnet'  # TODO: other source
        self.lang = lang
        self.model = self.get_model(self.aug_src, lang)

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
    def get_model(cls, aug_src, lang):
        if aug_src == 'wordnet':
            return nmw.WordNet(lang=lang, is_synonym=False)
