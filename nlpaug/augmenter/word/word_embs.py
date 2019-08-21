"""
    Augmenter that apply operation to textual input based on word embeddings.
"""

from nlpaug.augmenter.word import WordAugmenter
from nlpaug.util import Action
import nlpaug.model.word_embs as nmw

WORD2VEC_MODEL = None
GLOVE_MODEL = None
FASTTEXT_MODEL = None
model_types = ['word2vec', 'glove', 'fasttext']


def init_word2vec_model(model_path, force_reload=False):
    # Load model once at runtime
    global WORD2VEC_MODEL
    if WORD2VEC_MODEL and not force_reload:
        return WORD2VEC_MODEL

    word2vec = nmw.Word2vec()
    word2vec.read(model_path)
    WORD2VEC_MODEL = word2vec

    return WORD2VEC_MODEL


def init_glove_model(model_path, force_reload=False):
    # Load model once at runtime
    global GLOVE_MODEL
    if GLOVE_MODEL and not force_reload:
        return GLOVE_MODEL

    glove = nmw.GloVe()
    glove.read(model_path)
    GLOVE_MODEL = glove

    return GLOVE_MODEL


def init_fasttext_model(model_path, force_reload=False):
    # Load model once at runtime
    global FASTTEXT_MODEL
    if FASTTEXT_MODEL and not force_reload:
        return FASTTEXT_MODEL

    fasttext = nmw.Fasttext()
    fasttext.read(model_path)
    FASTTEXT_MODEL = fasttext

    return FASTTEXT_MODEL


class WordEmbsAug(WordAugmenter):
    """
    Augmenter that leverage word embeddings to find top n similar word for augmentation.

    :param str model_type: Model type of word embeddings. Expected values include 'word2vec', 'glove' and 'fasttext'.
    :param str model_path: Downloaded model directory. Either model_path or model is must be provided
    :param obj model: Pre-loaded model
    :param str action: Either 'insert or 'substitute'. If value is 'insert', a new word will be injected to random
        position according to word embeddings calculation. If value is 'substitute', word will be replaced according
        to word embeddings calculation
    :param int aug_min: Minimum number of word will be augmented.
    :param float aug_p: Percentage of word will be augmented.
    :param int aug_n: Top n similar word for lucky draw
    :param list stopwords: List of words which will be skipped from augment operation.
    :param func tokenizer: Customize tokenization process
    :param func reverse_tokenizer: Customize reverse of tokenization process
    :param bool force_reload: If True, model will be loaded every time while it takes longer time for initialization.
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.word as naw
    >>> aug = naw.WordEmbsAug(model_type='word2vec', model_path='.')
    """

    def __init__(self, model_type, model_path='.', model=None, action=Action.SUBSTITUTE,
                 name='WordEmbs_Aug', aug_min=1, aug_p=0.3, aug_n=5, n_gram_separator='_',
                 stopwords=None, tokenizer=None, reverse_tokenizer=None, force_reload=False, verbose=0):
        super().__init__(
            action=action, name=name, aug_p=aug_p, aug_min=aug_min, stopwords=stopwords,
            tokenizer=tokenizer, reverse_tokenizer=reverse_tokenizer, verbose=verbose)

        self.model_type = model_type
        self.model_path = model_path
        self.aug_n = aug_n
        self.n_gram_separator = n_gram_separator

        self.pre_validate()

        if model is None:
            self.model = self.get_model(model_type=model_type, force_reload=force_reload)
        else:
            self.model = model

    def pre_validate(self):
        if self.model_type not in model_types:
            raise ValueError('Model type value is unexpected. Expected values include {}'.format(model_types))

    def get_model(self, model_type, force_reload=False):
        if model_type == 'word2vec':
            return init_word2vec_model(self.model_path, force_reload)
        elif model_type == 'glove':
            return init_glove_model(self.model_path, force_reload)
        elif model_type == 'fasttext':
            return init_fasttext_model(self.model_path, force_reload)
        else:
            return None

    def skip_aug(self, token_idxes, tokens):
        results = []
        for token_idx in token_idxes:
            # Some words do not come with vector. It will be excluded in lucky draw.
            word = tokens[token_idx]

            if word in self.model.w2v:
                results.append(token_idx)

        return results

    def insert(self, data):
        tokens = self.tokenizer(data)
        results = tokens.copy()

        aug_idexes = self._get_random_aug_idxes(tokens)
        if aug_idexes is None:
            return data
        aug_idexes.sort(reverse=True)

        for aug_idx in aug_idexes:
            new_word = self.sample(self.model.get_vocab(), 1)[0]
            if self.n_gram_separator in new_word:
                new_word = new_word.split(self.n_gram_separator)[0]
            results.insert(aug_idx, new_word)

        return self.reverse_tokenizer(results)

    def substitute(self, data):
        tokens = self.tokenizer(data)
        results = tokens.copy()

        aug_idexes = self._get_aug_idxes(tokens)
        if aug_idexes is None:
            return data

        for aug_idx in aug_idexes:
            original_word = results[aug_idx]
            candidate_words = self.model.predict(original_word, top_n=self.aug_n)
            substitute_word = self.sample(candidate_words, 1)[0]

            results[aug_idx] = substitute_word

        return self.reverse_tokenizer(results)
