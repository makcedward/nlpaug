import logging

try:
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline
except ImportError:
    # No installation required if not using this function
    pass

from nlpaug.model.lang_models import LanguageModels
from nlpaug.util.selection.filtering import *


class Transformers(LanguageModels):
    def __init__(self, model_path='bert-base-uncased', model_type='bert', top_k=None, device='cuda', silence=True):
        super().__init__(device, model_type=model_type, top_k=top_k, silence=silence)
        try:
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ModuleNotFoundError:
            raise ModuleNotFoundError('Missed transformers library. Install transfomers by `pip install transformers`')

        self.model_path = model_path
        if silence:
            # Transformers thrown an warning regrading to weight initialization. It is expected
            orig_log_level = logging.getLogger('transformers.' + 'modeling_utils').getEffectiveLevel()
            logging.getLogger('transformers.' + 'modeling_utils').setLevel(logging.ERROR)

            if device == 'cpu' or device is None:
                device = -1
            elif device == 'cuda':
                device = 0
            elif 'cuda:' in device:
                device = int(device.split(':')[1])

            if top_k is None:
                top_k = 5

            self.model = pipeline("fill-mask", model=model_path, device=device, top_k=top_k)
            logging.getLogger('transformers.' + 'modeling_utils').setLevel(orig_log_level)

    def get_tokenizer(self):
        return self.model.tokenizer

    def get_model(self):
        return self.model.model

    def get_max_num_token(self):
        return self.model.model.config.max_position_embeddings - 2 * 5

    def is_skip_candidate(self, candidate):
        return candidate.startswith(self.get_subword_prefix())

    def token2id(self, token):
        # Iseue 181: TokenizerFast have convert_tokens_to_ids but not convert_tokens_to_id
        if 'TokenizerFast' in self.tokenizer.__class__.__name__:
            # New transformers API
            return self.model.tokenizer.convert_tokens_to_ids(token)
        else:
            # Old transformers API
            return self.model.tokenizer._convert_token_to_id(token)

    def id2token(self, _id):
        return self.model.tokenizer._convert_id_to_token(_id)

    def predict(self, texts, target_words=None, n=1):
        results = [] 
        for text in texts:
            result = self.model(text)
            results.append([r['token_str'] for r in result])

        return results
