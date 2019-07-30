import re

from nlpaug.augmenter.word import WordAugmenter
from nlpaug.util import Action, Warning, WarningName, WarningCode, WarningMessage
import nlpaug.model.word_stats as nmws

TFIDF_MODEL = {}

def init_tfidf_model(model_path, force_reload=False):
    """
        Load model once at runtime
    """
    global TFIDF_MODEL
    if TFIDF_MODEL and not force_reload:
        return TFIDF_MODEL

    tfidf_model = nmws.TfIdf()
    tfidf_model.read(model_path)
    TFIDF_MODEL = tfidf_model

    return tfidf_model


class TfIdfAug(WordAugmenter):
    def __init__(self, model_path='.', action=Action.SUBSTITUTE,
                 name='TfIdf_Aug', aug_min=1, aug_p=0.3, aug_n=5, n_gram_separator='_',
                 stopwords=[], tokenizer=None, reverse_tokenizer=None, verbose=0):
        super().__init__(
            action=action, name=name, aug_p=aug_p, aug_min=aug_min, stopwords=stopwords,
            tokenizer=tokenizer, reverse_tokenizer=reverse_tokenizer, verbose=verbose)
        self.model_path = model_path
        self.aug_n = aug_n
        self.model = self.get_model(force_reload=False)
        self.n_gram_separator = n_gram_separator

    def skip_aug(self, token_idxes, tokens):
        results = []
        for token_idx in token_idxes:
            """
                Some word does not come with IDF. It will be excluded in lucky draw. 
            """
            word = tokens[token_idx]

            if word in self.model.w2idf:
                results.append(token_idx)

        return results

    def _get_aug_idxes(self, tokens):
        aug_cnt = self.generate_aug_cnt(len(tokens))
        word_idxes = [i for i, t in enumerate(tokens) if t not in self.stopwords]
        word_idxes = self.skip_aug(word_idxes, tokens)

        if len(word_idxes) == 0:
            if self.verbose > 0:
                exception = Warning(name=WarningName.OUT_OF_VOCABULARY,
                                    code=WarningCode.WARNING_CODE_002, msg=WarningMessage.NO_WORD)
                exception.output()
            return None
        if len(word_idxes) < aug_cnt:
            aug_cnt = len(word_idxes)

        aug_probs = self.model.cal_tfidf(word_idxes, tokens)
        aug_idxes = []

        # It is possible that no token is picked. So re-try
        retry_cnt = 3
        possible_idxes = word_idxes.copy()
        for _ in range(retry_cnt):
            for i, p in zip(possible_idxes, aug_probs):
                if self.prob() < p:
                    aug_idxes.append(i)
                    possible_idxes.remove(i)

                    if len(possible_idxes) == aug_cnt:
                        break

        # If still cannot pick up, random pick index regrardless probability
        if len(aug_idxes) < aug_cnt:
            aug_idxes.extend(self.sample(possible_idxes, aug_cnt-len(aug_idxes)))

        aug_idxes = self.sample(aug_idxes, aug_cnt)

        return aug_idxes

    def insert(self, text):
        tokens = self.tokenizer(text)
        results = tokens.copy()

        aug_idxes = self._get_random_aug_idxes(tokens)
        aug_idxes.sort(reverse=True)

        for aug_idx in aug_idxes:
            original_word = results[aug_idx]
            new_word = self.sample(self.model.predict(original_word, top_n=self.aug_n), 1)[0]
            results.insert(aug_idx, new_word)

        return self.reverse_tokenizer(results)

    def substitute(self, text):
        tokens = self.tokenizer(text)
        results = tokens.copy()

        aug_idxes = self._get_aug_idxes(tokens)
        if aug_idxes is None:
            return text

        for aug_idx in aug_idxes:
            original_word = results[aug_idx]
            candidate_words = self.model.predict(original_word, top_n=self.aug_n)
            substitute_word = self.sample(candidate_words, 1)[0]

            results[aug_idx] = substitute_word

        return self.reverse_tokenizer(results)

    def get_model(self, force_reload=False):
        return init_tfidf_model(self.model_path, force_reload)

