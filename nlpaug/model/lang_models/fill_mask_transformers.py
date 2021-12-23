import logging

try:
    import torch
    from transformers import pipeline
except ImportError:
    # No installation required if not using this function
    pass

from nlpaug.model.lang_models import LanguageModels


class FmTransformers(LanguageModels):
    def __init__(self, model_path='bert-base-uncased', model_type='bert', top_k=None, device='cuda', 
        max_length=300, batch_size=32, silence=True):
        super().__init__(device, model_type=model_type, top_k=top_k, silence=silence)
        try:
            from transformers import pipeline
        except ModuleNotFoundError:
            raise ModuleNotFoundError('Missed transformers library. Install transfomers by `pip install transformers`')

        self.max_length = max_length
        self.batch_size = batch_size
        self.model_path = model_path
        device = self.convert_device(device)
        top_k = top_k if top_k else 5

        if silence:
            # Transformers thrown an warning regrading to weight initialization. It is expected
            orig_log_level = logging.getLogger('transformers.' + 'modeling_utils').getEffectiveLevel()
            logging.getLogger('transformers.' + 'modeling_utils').setLevel(logging.ERROR)
            self.model = pipeline("fill-mask", model=model_path, device=device, top_k=top_k)
            logging.getLogger('transformers.' + 'modeling_utils').setLevel(orig_log_level)
        else:
            self.model = pipeline("fill-mask", model=model_path, device=device, top_k=top_k)

    def to(self, device):
        self.model.model.to(device)

    def get_device(self):
        return str(self.model.device)

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

        predict_results = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                predict_result = self.model(texts[i:i+self.batch_size], num_workers=1)
                if isinstance(predict_result, list) and len(predict_result) > 0:
                    if isinstance(predict_result[0], list):
                        predict_results.extend(predict_result)
                    else:
                        predict_results.extend([predict_result])

        for result in predict_results:
            temp_results = []
            for r in result:
                token = r['token_str']
                if self.model_type in ['bert'] and token.startswith('##'):
                    continue
                # subword came without space for roberta but not normal subowrd prefix
                if self.model_type in ['roberta', 'bart'] and not token.startswith(' '):
                    continue

                temp_results.append(token)

            results.append(temp_results)
    
        return results
