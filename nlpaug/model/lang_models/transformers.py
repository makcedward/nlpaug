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
    START_TOKEN = '[CLS]'
    SEPARATOR_TOKEN = '[SEP]'
    MASK_TOKEN = '[MASK]'
    PAD_TOKEN = '[PAD]'
    UNKNOWN_TOKEN = '[UNK]'
    SUBWORD_PREFIX = '##'

    def __init__(self, model_path='bert-base-uncased', temperature=1.0, top_k=None, top_p=None, device='cuda', silence=True):
        super().__init__(device, temperature=temperature, top_k=top_k, top_p=top_p, silence=silence)
        try:
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ModuleNotFoundError:
            raise ModuleNotFoundError('Missed transformers library. Install transfomers by `pip install transformers`')

        self.model_path = model_path

        # self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # self.mask_id = self.token2id(self.MASK_TOKEN)
        # self.pad_id = self.token2id(self.PAD_TOKEN)
        if silence:
            # Transformers thrown an warning regrading to weight initialization. It is expected
            orig_log_level = logging.getLogger('transformers.' + 'modeling_utils').getEffectiveLevel()
            logging.getLogger('transformers.' + 'modeling_utils').setLevel(logging.ERROR)
            self.model = pipeline("fill-mask", model=model_path, device=device)
            logging.getLogger('transformers.' + 'modeling_utils').setLevel(orig_log_level)

    def get_max_num_token(self):
        return self.model.model.config.max_position_embeddings - 2 * 5

    def is_skip_candidate(self, candidate):
        return candidate.startswith(self.SUBWORD_PREFIX)

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

