"""
    Augmenter that apply fasttext's based operation to textual input.
"""

from nlpaug.augmenter.word import WordEmbsAugmenter
from nlpaug.util import Action
import nlpaug.model.word_embs as nmw
from nlpaug.util.decorator.deprecation import deprecated

FASTTEXT_MODEL = {}


def init_fasttext_model(model_path, force_reload=False):
    """
        Load model once at runtime
    """
    global FASTTEXT_MODEL
    if FASTTEXT_MODEL and not force_reload:
        return FASTTEXT_MODEL

    fasttext = nmw.Fasttext()
    fasttext.read(model_path)
    FASTTEXT_MODEL = fasttext

    return FASTTEXT_MODEL


@deprecated(deprecate_from='0.0.7', deprecate_to='0.0.9', msg="Use WordEmbsAug from 0.0.7 version")
class FasttextAug(WordEmbsAugmenter):
    """
    Augmenter that leverage fasttext's embeddings to find top n similar word for augmentation.

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
    >>> aug = naw.FasttextAug()
    """

    def __init__(self, model_path='.', model=None, action=Action.SUBSTITUTE,
                 name='Fasttext_Aug', aug_min=1, aug_p=0.3, aug_n=5, stopwords=None,
                 tokenizer=None, reverse_tokenizer=None, force_reload=False,
                 verbose=0):
        super().__init__(
            model_path=model_path, aug_n=aug_n,
            action=action, name=name, aug_p=aug_p, aug_min=aug_min, stopwords=stopwords,
            tokenizer=tokenizer, reverse_tokenizer=reverse_tokenizer, verbose=verbose)

        if model is None:
            self.model = self.get_model(force_reload=force_reload)
        else:
            self.model = model

    def get_model(self, force_reload=False):
        return init_fasttext_model(self.model_path, force_reload)
