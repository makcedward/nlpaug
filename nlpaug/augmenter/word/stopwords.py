"""
    Augmenter that apply stopwords removal to textual input.
"""

from nlpaug.augmenter.word import WordAugmenter
from nlpaug.util import Action, Warning, WarningName, WarningCode, WarningMessage
from nlpaug.util.decorator.deprecation import deprecated


@deprecated(deprecate_from='0.0.7', deprecate_to='0.0.9', msg="Use RandomWordAug from 0.0.7 version")
class StopWordsAug(WordAugmenter):
    """
    Augmenter that leverage pre-defined spelling mistake dictionary to simulate spelling mistake.

    :param list stopwords: List of stopword will be removed.
    :param int aug_min: Minimum number of word will be augmented.
    :param float aug_p: Percentage of word will be augmented.
    :param func tokenizer: Customize tokenization process
    :param func reverse_tokenizer: Customize reverse of tokenization process
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.word as naw
    >>> aug = naw.StopWordsAug(stopwords=['a', 'an', 'the'])
    """

    def __init__(self, stopwords, name='StopWords_Aug', aug_min=1, aug_p=0.3,
                 tokenizer=None, reverse_tokenizer=None, case_sensitive=False, verbose=0):

        if not case_sensitive:
            stopwords = [t.lower() for t in stopwords]

        super().__init__(
            action=Action.DELETE, name=name, aug_p=aug_p, aug_min=aug_min, stopwords=stopwords,
            tokenizer=tokenizer, reverse_tokenizer=reverse_tokenizer, verbose=verbose)

        self.case_sensitive = case_sensitive

    def _get_aug_idxes(self, tokens):
        aug_cnt = self.generate_aug_cnt(len(tokens))
        word_idxes = []
        for i, t in enumerate(tokens):
            token = t
            if not self.case_sensitive:
                token = token.lower()
            if token in self.stopwords:
                word_idxes.append(i)

        word_idxes = self.skip_aug(word_idxes, tokens)
        if len(word_idxes) == 0:
            if self.verbose > 0:
                exception = Warning(name=WarningName.OUT_OF_VOCABULARY,
                                    code=WarningCode.WARNING_CODE_002, msg=WarningMessage.NO_WORD)
                exception.output()
            return None
        if len(word_idxes) < aug_cnt:
            aug_cnt = len(word_idxes)
        aug_idexes = self.sample(word_idxes, aug_cnt)
        return aug_idexes

    def delete(self, text):
        tokens = self.tokenizer(text)
        results = tokens.copy()

        aug_idxes = self._get_aug_idxes(tokens)
        if aug_idxes is None:
            return text

        aug_idxes.sort(reverse=True)

        for aug_idx in aug_idxes:
            del results[aug_idx]

        if len(results) > 0 and len(results[0]) > 0:
            results[0] = self.align_capitalization(tokens[0], results[0])

        return self.reverse_tokenizer(results)
