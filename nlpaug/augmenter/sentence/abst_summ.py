"""
    Augmenter that apply operation (sentence level) to textual input based on abstractive summarization.
"""

import os

from nlpaug.augmenter.sentence import SentenceAugmenter
import nlpaug.model.lang_models as nml
from nlpaug.util import Action, Doc

ABST_SUMM_MODELS = {}

def init_abst_summ_model(model_path, min_length, max_length, num_beam, no_repeat_ngram_size, device, 
    force_reload=False):
    global ABST_SUMM_MODELS

    model_name = os.path.basename(model_path)
    if model_name in ABST_SUMM_MODELS and not force_reload:
        ABST_SUMM_MODELS[model_name].min_length = min_length
        ABST_SUMM_MODELS[model_name].max_length = max_length
        ABST_SUMM_MODELS[model_name].num_beam = num_beam
        ABST_SUMM_MODELS[model_name].no_repeat_ngram_size = no_repeat_ngram_size
        ABST_SUMM_MODELS[model_name].device = device
        return ABST_SUMM_MODELS[model_name]

    if 't5' in model_path:
        model = nml.T5(model_path, device=device, min_length=min_length, max_length=max_length,
            num_beam=num_beam, no_repeat_ngram_size=no_repeat_ngram_size)
    elif 'bart-large-cnn' in model_path:
        model = nml.Bart(model_path, device=device, min_length=min_length, max_length=max_length,
            num_beam=num_beam, no_repeat_ngram_size=no_repeat_ngram_size)
    else:
        raise ValueError('Model name value is unexpected. Only support T5 model.')

    ABST_SUMM_MODELS[model_name] = model
    return model


class AbstSummAug(SentenceAugmenter):

    """
    Augmenter that leverage contextual word embeddings to find top n similar word for augmentation.

    :param str model_path: Model name or model path. It used transformers to load the model. Tested 'facebook/bart-large-cnn',
        t5-small', 't5-base' and 't5-large'
    :param float min_length: The minium output length of result. If it is less than 1, it will calculated as length 
        of input times this value. For example, input length is 100 (character length, not nubmer of token) while 
        min_length is 0.3. The minium output length is 30 (100 * 0.3). Default value is 10.
    :param float max_length: The maximum output length of result. If it is less than 1, it will calculated as length 
        of input times this value. For example, input length is 100 (character length, not nubmer of token) while 
        max_length is 0.3. The maximum output length is 30 (100 * 0.3). Default value is 50. If max_length is larger
        or equal to length of input. The maximum length becomes length of ipnut * 0.5.
    :param int num_beam: Number of beams for beam search in summarization. No beam search if it is 1. Default is 3.
    :param int no_repeat_ngram_size: If value is 0, all ngrams of that size can only occur once. Default value is 3
    :param str device: Default value is CPU. If value is CPU, it uses CPU for processing. If value is CUDA, it uses GPU
        for processing. Possible values include 'cuda' and 'cpu'. (May able to use other options)
    :param bool force_reload: Force reload the contextual word embeddings model to memory when initialize the class.
        Default value is False and suggesting to keep it as False if performance is the consideration.
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.sentence as nas
    >>> aug = nas.AbstSummAug()
    """

    def __init__(self, model_path='t5-base', min_length=10, max_length=50, num_beam=3, no_repeat_ngram_size=3, 
        name='AbstSumm_Aug', device='cpu', force_reload=False, verbose=0):
        super().__init__(
            action=Action.SUBSTITUTE, name=name, tokenizer=None, stopwords=None, device=device,
            include_detail=False, verbose=verbose, parallelable=True)
        self.model_path = model_path

        self._init()
        self.model = self.get_model(
            model_path=model_path, device=device, force_reload=force_reload, min_length=min_length, 
            max_length=max_length, num_beam=num_beam, no_repeat_ngram_size=no_repeat_ngram_size)
        self.device = self.model.device

    def _init(self):
        if 't5' in self.model_path:
            self.model_type = 't5'
        elif 'bart-large-cnn' in self.model_path:
            self.model_type = 'bart'
        else:
            self.model_type = ''

    def substitute(self, data):
        if not data:
            return data

        if isinstance(data, list):
            all_data = data
        else:
            if data.strip() == '':
                return data
            all_data = [data]

        return self.model.predict(all_data)

    @classmethod
    def get_model(cls, model_path, min_length, max_length, num_beam, no_repeat_ngram_size, 
        device='cuda', force_reload=False):
        return init_abst_summ_model(model_path, min_length, max_length, num_beam, no_repeat_ngram_size,
            device, force_reload)
