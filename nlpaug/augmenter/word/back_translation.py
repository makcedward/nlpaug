"""
    Augmenter that apply operation (word level) to textual input based on back translation.
"""

import string
import os

from nlpaug.augmenter.word import WordAugmenter
import nlpaug.model.lang_models as nml

BACK_TRANSLATION_MODELS = {}


def init_back_translatoin_model(from_model_name, from_model_checkpt, to_model_name, to_model_checkpt, 
        tokenzier_name, bpe_name, is_load_from_github, device, force_reload=False):
    global BACK_TRANSLATION_MODELS

    model_name = '_'.join([from_model_name, to_model_name])
    if model_name in BACK_TRANSLATION_MODELS and not force_reload:
        BACK_TRANSLATION_MODELS[model_name].tokenzier_name = tokenzier_name
        BACK_TRANSLATION_MODELS[model_name].bpe_name = bpe_name
        BACK_TRANSLATION_MODELS[model_name].is_load_from_github = is_load_from_github
        BACK_TRANSLATION_MODELS[model_name].device = device

        return BACK_TRANSLATION_MODELS[model_name]
    model = nml.Fairseq(from_model_name=from_model_name, from_model_checkpt=from_model_checkpt, 
        to_model_name=to_model_name, to_model_checkpt=to_model_checkpt, 
        tokenzier_name=tokenzier_name, bpe_name=bpe_name, is_load_from_github=is_load_from_github, 
        device=device)

    BACK_TRANSLATION_MODELS[model_name] = model
    return model


class BackTranslationAug(WordAugmenter):
    # https://arxiv.org/pdf/1511.06709.pdf
    """
    Augmenter that leverage two translation models for augmentation. For example, the source is English. This
    augmenter translate source to German and translating it back to English. For detail, you may visit
    https://towardsdatascience.com/data-augmentation-in-nlp-2801a34dfc28

    :param str from_model_name: Language of your text. Veriried 'transformer.wmt18.en-de', 'transformer.wmt19.en-de', 
        'transformer.wmt19.de-en', 'transformer.wmt19.en-ru' and 'transformer.wmt19.ru-en'
    :param str to_model_name: Language for translation. Veriried 'transformer.wmt18.en-de', transformer.wmt19.en-de', 
        'transformer.wmt19.de-en', 'transformer.wmt19.en-ru' and 'transformer.wmt19.ru-en'
    :param str tokenizer: Default value is 'moses'
    :param str bpe: Default value is 'fastbpe'
    :param str device: Default value is CPU. If value is CPU, it uses CPU for processing. If value is CUDA, it uses GPU
        for processing. Possible values include 'cuda' and 'cpu'. (May able to use other options)
    :param bool is_load_from_github: Default is True. If True, transaltion models will be loaded from fairseq's
        github. Otherwise, providing model directory for both `from_model_name` and `to_model_name` parameters.
    :param bool force_reload: Force reload the contextual word embeddings model to memory when initialize the class.
        Default value is False and suggesting to keep it as False if performance is the consideration.
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.word as naw
    >>> aug = naw.BackTranslationAug()
    """

    def __init__(self, from_model_name='transformer.wmt19.en-de', to_model_name='transformer.wmt19.de-en', 
        from_model_checkpt='model1.pt', to_model_checkpt='model1.pt', tokenizer='moses', bpe='fastbpe', 
        is_load_from_github=True, name='BackTranslationAug', device='cpu', force_reload=False, verbose=0):
        super().__init__(
            action='substitute', name=name, aug_p=None, aug_min=None, aug_max=None, tokenizer=None, 
            device=device, verbose=verbose, include_detail=False, parallelable=True)

        self.model = self.get_model(
            from_model_name=from_model_name, from_model_checkpt=from_model_checkpt, 
            to_model_name=to_model_name, to_model_checkpt=to_model_checkpt, 
            tokenzier_name=tokenizer, bpe_name=bpe, device=device,
            is_load_from_github=is_load_from_github
        )
        self.device = self.model.device
        self.is_load_from_github = is_load_from_github

    def substitute(self, data):
        if not data:
            return data
            
        augmented_text = self.model.predict(data)
        return augmented_text

    @classmethod
    def get_model(cls, from_model_name, from_model_checkpt, to_model_name, to_model_checkpt, 
            tokenzier_name, bpe_name, device='cuda', is_load_from_github=True, force_reload=False):
        return init_back_translatoin_model(from_model_name, from_model_checkpt, 
            to_model_name, to_model_checkpt, tokenzier_name, bpe_name,
            is_load_from_github, device, force_reload
        )

    @classmethod
    def clear_cache(cls):
        global BACK_TRANSLATION_MODELS
        BACK_TRANSLATION_MODELS = {}
