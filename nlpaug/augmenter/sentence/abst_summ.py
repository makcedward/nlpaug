"""
    Augmenter that apply operation (sentence level) to textual input based on abstractive summarization.
"""

import os

from nlpaug.augmenter.sentence import SentenceAugmenter
import nlpaug.model.lang_models as nml
from nlpaug.util import Action, Doc

ABST_SUMM_MODELS = {}

def init_abst_summ_model(model_path, tokenizer_path, device, force_reload=False,
    min_length=20, max_length=50, batch_size=32, temperature=1.0, top_k=50, top_p=0.9, 
    use_custom_api=True):
    global ABST_SUMM_MODELS

    model_name = '_'.join([os.path.basename(model_path), os.path.basename(tokenizer_path), str(device)])
    if model_name in ABST_SUMM_MODELS and not force_reload:
        ABST_SUMM_MODELS[model_name].min_length = min_length
        ABST_SUMM_MODELS[model_name].max_length = max_length
        ABST_SUMM_MODELS[model_name].temperature = temperature
        ABST_SUMM_MODELS[model_name].top_k = top_k
        ABST_SUMM_MODELS[model_name].top_p = top_p
        ABST_SUMM_MODELS[model_name].batch_size = batch_size
        return ABST_SUMM_MODELS[model_name]

    if use_custom_api:
        num_beam = 3
        no_repeat_ngram_size = 3
        
        if 't5' in model_path:
            model = nml.T5(model_path, device=device, min_length=min_length, max_length=max_length,
                num_beam=num_beam, no_repeat_ngram_size=no_repeat_ngram_size)
        elif 'bart-large-cnn' in model_path:
            model = nml.Bart(model_path, device=device, min_length=min_length, max_length=max_length,
                num_beam=num_beam, no_repeat_ngram_size=no_repeat_ngram_size)
        else:
            raise ValueError('Model name value is unexpected. Only support `T5` and `bart-large-cnn` model.')
    else:
        model = nml.XSumTransformers(model_name=model_path, tokenizer_name=tokenizer_path, 
            min_length=min_length, max_length=max_length, temperature=temperature, top_k=top_k,
            top_p=top_p, batch_size=batch_size, device=device)

    ABST_SUMM_MODELS[model_name] = model
    return model

class AbstSummAug(SentenceAugmenter):

    """
    Augmenter that leverage contextual word embeddings to find top n similar word for augmentation.

    :param str model_path: Model name or model path. It used transformers to load the model. Tested 'facebook/bart-large-cnn',
        t5-small', 't5-base' and 't5-large'. For models, you can visit https://huggingface.co/models?filter=summarization
    :param int batch_size: Batch size.
    :param int min_length: The min length of output text.
    :param int max_length: The max length of output text. 
    :param float temperature: The value used to module the next token probabilities.
    :param int top_k: The number of highest probability vocabulary tokens to keep for top-k-filtering.
    :param float top_p: If set to float < 1, only the most probable tokens with probabilities that add up to `top_p` or
        higher are kept for generation.
    :param str device: Default value is CPU. If value is CPU, it uses CPU for processing. If value is CUDA, it uses GPU
        for processing. Possible values include 'cuda' and 'cpu'. (May able to use other options)
    :param bool force_reload: Force reload the contextual word embeddings model to memory when initialize the class.
        Default value is False and suggesting to keep it as False if performance is the consideration.
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.sentence as nas
    >>> aug = nas.AbstSummAug()
    """

    def __init__(self, model_path='t5-base', tokenizer_path='t5-base', 
        min_length=20, max_length=50, batch_size=32, temperature=1.0, top_k=50, top_p=0.9, 
        name='AbstSumm_Aug', device='cpu', force_reload=False, verbose=0, use_custom_api=True):
        super().__init__(
            action=Action.SUBSTITUTE, name=name, tokenizer=None, stopwords=None, device=device,
            include_detail=False, verbose=verbose)
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path

        self.model = self.get_model(
            model_path=model_path, tokenizer_path=tokenizer_path, device=device, force_reload=force_reload, 
            min_length=min_length, max_length=max_length, batch_size=batch_size, temperature=temperature,
            top_k=top_k, top_p=top_p, use_custom_api=use_custom_api)
        self.device = self.model.device

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
    def get_model(cls, model_path, tokenizer_path, device='cuda', force_reload=False, 
        min_length=20, max_length=50, batch_size=32, temperature=1.0, top_k=50, top_p=0.9, 
        use_custom_api=True):
        return init_abst_summ_model(model_path, tokenizer_path, device, force_reload, 
            min_length, max_length, batch_size, temperature, top_k, top_p, use_custom_api)
