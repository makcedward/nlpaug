import logging

try:
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
except ImportError:
    # No installation required if not using this function
    pass

from nlpaug.model.lang_models import LanguageModels
from nlpaug.util.selection.filtering import *


class DistilBert(LanguageModels):
    # https://arxiv.org/pdf/1910.01108.pdf
    START_TOKEN = '[CLS]'
    SEPARATOR_TOKEN = '[SEP]'
    MASK_TOKEN = '[MASK]'
    PAD_TOKEN = '[PAD]'
    UNKNOWN_TOKEN = '[UNK]'
    SUBWORD_PREFIX = '##'

    def __init__(self, model_path='distilbert-base-uncased', temperature=1.0, top_k=None, top_p=None, batch_size=32,
        device='cuda', silence=True):
        super().__init__(device, temperature=temperature, top_k=top_k, top_p=top_p, batch_size=batch_size, silence=True)
        try:
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ModuleNotFoundError:
            raise ModuleNotFoundError('Missed transformers library. Install transfomers by `pip install transformers`')
            
        self.model_path = model_path

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.mask_id = self.token2id(self.MASK_TOKEN)
        self.pad_id = self.token2id(self.PAD_TOKEN)
        if silence:
            # Transformers thrown an warning regrading to weight initialization. It is expected
            orig_log_level = logging.getLogger('transformers.' + 'modeling_utils').getEffectiveLevel()
            logging.getLogger('transformers.' + 'modeling_utils').setLevel(logging.ERROR)
            self.model = AutoModelForMaskedLM.from_pretrained(model_path)
            logging.getLogger('transformers.' + 'modeling_utils').setLevel(orig_log_level)
        else:
            self.model = AutoModelForMaskedLM.from_pretrained(model_path)

        self.model.to(self.device)
        self.model.eval()

    def get_max_num_token(self):
        return self.model.config.max_position_embeddings - 2 * 5

    def is_skip_candidate(self, candidate):
        return candidate[:2] == self.SUBWORD_PREFIX

    def token2id(self, token):
        # Iseue 181: TokenizerFast have convert_tokens_to_ids but not convert_tokens_to_id
        if 'TokenizerFast' in self.tokenizer.__class__.__name__:
            # New transformers API
            return self.tokenizer.convert_tokens_to_ids(token)
        else:
            # Old transformers API
            return self.tokenizer._convert_token_to_id(token)

    def id2token(self, _id):
        return self.tokenizer._convert_id_to_token(_id)

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

    def get_subword_prefix(self):
        return self.SUBWORD_PREFIX

    def get_mask_token(self):
        return self.MASK_TOKEN

    def predict(self, texts, target_words=None, n=1):
        results = []
        # Prepare inputs
        for i in range(0, len(texts), self.batch_size):
            token_inputs = [self.tokenizer.encode(text) for text in texts[i:i+self.batch_size]]
            if target_words is None:
                target_words = [None] * len(token_inputs)
            
            # Pad token
            max_token_size = max([len(t) for t in token_inputs])
            for i, token_input in enumerate(token_inputs):
                for _ in range(max_token_size - len(token_input)):
                    token_inputs[i].append(self.pad_id)
            
            target_poses = []
            for tokens in token_inputs:
                target_poses.append(tokens.index(self.mask_id))
            mask_inputs = [[1] * len(tokens) for tokens in token_inputs] # 1: real token, 0: padding token

            # Convert to feature
            token_inputs = torch.tensor(token_inputs).to(self.device)
            mask_inputs = torch.tensor(mask_inputs).to(self.device)

            # Prediction
            results = []
            with torch.no_grad():
                outputs = self.model(input_ids=token_inputs, attention_mask=mask_inputs)

            # Selection
            for output, target_pos, target_token in zip(outputs[0], target_poses, target_words):
                target_token_logits = output[target_pos]
                
                seed = {'temperature': self.temperature, 'top_k': self.top_k, 'top_p': self.top_p}
                target_token_logits = self.control_randomness(target_token_logits, seed)
                target_token_logits, target_token_idxes = self.filtering(target_token_logits, seed)
                if len(target_token_idxes) != 0:
                    new_tokens = self.pick(target_token_logits, target_token_idxes, target_word=target_token, n=10)
                    results.append([t[0] for t in new_tokens])
                else:
                    results.append([''])

        return results
