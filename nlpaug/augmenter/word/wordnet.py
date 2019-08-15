"""
    Augmenter that apply semantic meaning based to textual input.
"""

import nltk
from nltk.corpus import wordnet

from nlpaug.augmenter.word import WordAugmenter
from nlpaug.util import Action, PartOfSpeech, Warning, WarningName, WarningCode, WarningMessage


class WordNetAug(WordAugmenter):
    """
    Augmenter that leverage semantic meaning to substitute word.

    :param str lang: Language of your text. Default value is 'eng'.
    :param int aug_min: Minimum number of word will be augmented.
    :param float aug_p: Percentage of word will be augmented.
    :param list stopwords: List of words which will be skipped from augment operation.
    :param func tokenizer: Customize tokenization process
    :param func reverse_tokenizer: Customize reverse of tokenization process
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.word as naw
    >>> aug = naw.WordNetAug()
    """

    def __init__(self, name='WordNet_Aug', aug_min=1, aug_p=0.3, lang='eng', stopwords=[],
                 tokenizer=None, reverse_tokenizer=None, verbose=0):
        super().__init__(
            action=Action.SUBSTITUTE, name=name, aug_p=aug_p, aug_min=aug_min, stopwords=stopwords,
            tokenizer=tokenizer, reverse_tokenizer=reverse_tokenizer, verbose=verbose)

        self.model = self.get_model()
        self.lang = lang

        ### TODO: antonym: https://arxiv.org/pdf/1809.02079.pdf
        self.synonyms = True

    def skip_aug(self, token_idxes, tokens):
        results = []
        for token_idx in token_idxes:
            # Some word does not come with synonym. It will be excluded in lucky draw.
            if tokens[token_idx][1] not in ['DT']:
                results.append(token_idx)

        return results

    def _get_aug_idxes(self, tokens):
        aug_cnt = self.generate_aug_cnt(len(tokens))
        word_idxes = [i for i, t in enumerate(tokens) if t[0] not in self.stopwords]
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

    def substitute(self, text):
        results = []

        tokens = self.tokenizer(text)

        pos = nltk.pos_tag(tokens)

        aug_idxes = self._get_aug_idxes(pos)
        if aug_idxes is None:
            return text

        for i, token in enumerate(tokens):
            # Skip if no augment for word
            if i not in aug_idxes:
                results.append(token)
                continue

            word_poses = PartOfSpeech.pos2wn(pos[i][1])
            synets = []
            if word_poses is None or len(word_poses) == 0:
                # Use every possible words as the mapping does not defined correctly
                synets.extend(self.model.synsets(pos[i][0], lang=self.lang))
            else:
                for word_pos in word_poses:
                    synets.extend(self.model.synsets(pos[i][0], pos=word_pos, lang=self.lang))

            augmented_data = []
            for synet in synets:
                candidates = []
                for lema in synet.lemmas():
                    if self.synonyms:
                        candidates.append(lema.name())
                    else:
                        if lema.antonyms():
                            candidates.append(lema.antonyms()[0].name())

                for candidate in candidates:
                    if candidate.lower() != token.lower():
                        augmented_data.append(candidate)

            if len(augmented_data) == 0:
                results.append(token)
            else:
                candidate = self.sample(augmented_data, 1)[0]
                candidate = candidate.replace("_", " ").replace("-", " ").lower()
                results.append(self.align_capitalization(token, candidate))

        return self.reverse_tokenizer(results)

    def get_model(self):
        try:
            # Check whether wordnet package is downloaded
            wordnet.synsets('computer')
        except Exception:
            nltk.download('wordnet')

        try:
            # Check whether POS package is downloaded
            nltk.pos_tag('computer')
        except Exception:
            nltk.download('averaged_perceptron_tagger')

        return wordnet
