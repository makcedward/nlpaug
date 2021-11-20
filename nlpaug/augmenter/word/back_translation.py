"""
    Augmenter that apply operation (word level) to textual input based on back translation.
"""

import string
import os

from nlpaug.augmenter.word import WordAugmenter
import nlpaug.model.lang_models as nml

BACK_TRANSLATION_MODELS = {}


def init_back_translatoin_model(from_model_name, to_model_name, device, force_reload=False,
                                batch_size=32, max_length=None):
    global BACK_TRANSLATION_MODELS

    model_name = '_'.join([from_model_name, to_model_name, str(device)])
    if model_name in BACK_TRANSLATION_MODELS and not force_reload:
        BACK_TRANSLATION_MODELS[model_name].batch_size = batch_size
        BACK_TRANSLATION_MODELS[model_name].max_length = max_length

        return BACK_TRANSLATION_MODELS[model_name]

    model = nml.MtTransformers(src_model_name=from_model_name, tgt_model_name=to_model_name, 
        device=device, batch_size=batch_size, max_length=max_length)

    BACK_TRANSLATION_MODELS[model_name] = model
    return model


class BackTranslationAug(WordAugmenter):
    # https://arxiv.org/pdf/1511.06709.pdf
    """
    Augmenter that leverage two translation models for augmentation. For example, the source is English. This
    augmenter translate source to German and translating it back to English. For detail, you may visit
    https://towardsdatascience.com/data-augmentation-in-nlp-2801a34dfc28

    :param str from_model_name: Any model from https://huggingface.co/models?filter=translation&search=Helsinki-NLP. As
        long as from_model_name is pair with to_model_name. For example, from_model_name is English to Japanese,
        then to_model_name should be Japanese to English.
    :param str to_model_name: Any model from https://huggingface.co/models?filter=translation&search=Helsinki-NLP.
    :param str device: Default value is CPU. If value is CPU, it uses CPU for processing. If value is CUDA, it uses GPU
        for processing. Possible values include 'cuda' and 'cpu'. (May able to use other options)
    :param bool force_reload: Force reload the contextual word embeddings model to memory when initialize the class.
        Default value is False and suggesting to keep it as False if performance is the consideration.
    :param int batch_size: Batch size.
    :param int max_length: The max length of output text.
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.word as naw
    >>> aug = naw.BackTranslationAug()
    """

    def __init__(self, from_model_name='facebook/wmt19-en-de', to_model_name='facebook/wmt19-de-en',
        name='BackTranslationAug', device='cpu', batch_size=32, max_length=300, force_reload=False, verbose=0):
        super().__init__(
            action='substitute', name=name, aug_p=None, aug_min=None, aug_max=None, tokenizer=None,
            device=device, verbose=verbose, include_detail=False)

        self.model = self.get_model(from_model_name=from_model_name, to_model_name=to_model_name, 
            device=device, batch_size=batch_size, max_length=max_length
        )
        self.device = self.model.device

    def substitute(self, data, n=1):
        if not data:
            return data

        augmented_text = self.model.predict(data)
        return augmented_text

    @classmethod
    def get_model(cls, from_model_name, to_model_name, device='cuda', force_reload=False,
                  batch_size=32, max_length=None):
        return init_back_translatoin_model(from_model_name, to_model_name, device,
            force_reload, batch_size, max_length)

    @classmethod
    def clear_cache(cls):
        global BACK_TRANSLATION_MODELS
        BACK_TRANSLATION_MODELS = {}
