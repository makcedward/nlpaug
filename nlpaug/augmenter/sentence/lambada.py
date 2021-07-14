"""
    Augmenter that apply operation (sentence level) to textual input based on abstractive summarization.
"""

import os
import json

from nlpaug.augmenter.sentence import SentenceAugmenter
import nlpaug.model.lang_models as nml
from nlpaug.util import Action, Doc


class LambadaAug(SentenceAugmenter):

    """
    Augmenter that leverage contextual word embeddings to find top n similar word for augmentation.

    :param str model_dir: Directory of model. It is generated from train_lambada.sh under scritps folders.n
    :param float threshold: The threshold of classification probabilty for accpeting generated text. Return all result if threshold
        is None.
    :param int min_length: The min length of output text.
    :param int max_length: The max length of output text.
    :param int batch_size: Batch size.
    :param float temperature: The value used to module the next token probabilities.
    :param int top_k: The number of highest probability vocabulary tokens to keep for top-k-filtering.
    :param float top_p: If set to float < 1, only the most probable tokens with probabilities that add up to `top_p` or
        higher are kept for generation.
    :param float repetition_penalty : The parameter for repetition penalty. 1.0 means no penalty.
    :param str device: Default value is CPU. If value is CPU, it uses CPU for processing. If value is CUDA, it uses GPU
        for processing. Possible values include 'cuda' and 'cpu'. 
    :param bool force_reload: Force reload the contextual word embeddings model to memory when initialize the class.
        Default value is False and suggesting to keep it as False if performance is the consideration.
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.sentence as nas
    >>> aug = nas.LambadaAug()
    """

    def __init__(self, model_dir, threshold=None, min_length=100, max_length=300, 
        batch_size=16, temperature=1.0, top_k=50, top_p=0.9, repetition_penalty=1.0,
        name='Lambada_Aug', device='cpu', force_reload=False, verbose=0):
        super().__init__(
            action=Action.INSERT, name=name, tokenizer=None, stopwords=None, device=device,
            include_detail=False, verbose=verbose, parallelable=True)

        with open(os.path.join(model_dir, 'cls', 'label_encoder.json')) as json_file:
            self.label2id = json.load(json_file)
        
        self.model = nml.Lambada(
            cls_model_dir=os.path.join(model_dir, 'cls'), gen_model_dir=os.path.join(model_dir, 'gen'), 
            threshold=threshold, max_length=max_length, min_length=min_length, temperature=temperature, 
            top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, batch_size=batch_size,
            device=device)
        self.device = self.model.get_device()

    def insert(self, data, n=10):
        if not data:
            return data

        if isinstance(data, list):
            all_data = data
        else:
            if data.strip() == '':
                return data
            all_data = [data]

        for d in all_data:
            if d not in self.label2id:
                raise Exception('Label {} does not exist. Possible labels are {}'.format(d, self.label2id.keys()))

        return self.model.predict(all_data, n=n)
