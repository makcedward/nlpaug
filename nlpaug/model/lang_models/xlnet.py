import logging

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    # No installation required if not using this function
    pass

from nlpaug.model.lang_models import LanguageModels
from nlpaug.util.selection.filtering import *


# TODO: no optimize process yet
class XlNet(LanguageModels):
    # https://arxiv.org/abs/1906.08237

    # Since XLNet is not good on short inputs, Aman Rusia proposed to add padding text to overcome this limitation.
    # https://github.com/rusiaaman/XLNet-gen#methodology and https://github.com/huggingface/pytorch-transformers/issues/846
    PADDING_TEXT = """
        The quick brown fox jumps over the lazy dog. A horrible, messy split second presents
        itself to the heart-shaped version as Scott is moved. The upcoming movie benefits at 
        the mental cost of ages 14 to 12. Nothing substantial is happened for almost 48 days. 
        When that happens, we lose our heart. <eod>
    """
    MASK_TOKEN = '<mask>'

    NEW_PARAGRAPH_TOKEN = '<eop>'

    def __init__(self, model_path='xlnet-base-cased', top_k=None, padding_text=None,
                temperature=1.0, optimize=None, device=None, silence=True):
        super().__init__(device, model_type='xlnet', temperature=temperature, top_k=top_k, 
            top_p=None, optimize=optimize, silence=True)
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ModuleNotFoundError:
            raise ModuleNotFoundError('Missed transformers library. Install transfomers by `pip install transformers`')
            
        self.model_path = model_path

        # TODO: Evaluted to use mems in XLNet but the result is quite weird.
        self.optimize['external_memory'] = 0
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.mask_id = self.token2id(self.get_mask_token())
        self.pad_id = self.token2id(self.get_pad_token())
        config = {
            'mem_len': self.optimize['external_memory']
        }
        if silence:
            # Transformers thrown an warning regrading to weight initialization. It is expected
            orig_log_level = logging.getLogger('transformers.' + 'modeling_utils').getEffectiveLevel()
            logging.getLogger('transformers.' + 'modeling_utils').setLevel(logging.ERROR)
            self.model = AutoModelForCausalLM.from_pretrained(model_path, config=config)
            logging.getLogger('transformers.' + 'modeling_utils').setLevel(orig_log_level)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, config=config)

        self.padding_text_idxes = self.tokenizer.encode(padding_text or self.PADDING_TEXT)

        self.model.to(self.device)
        self.model.eval()

    def get_device(self):
        return str(self.model.device)

    def get_tokenizer(self):
        return self.tokenizer

    def get_model(self):
        return self.model

    def get_max_num_token(self):
        return 500

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

    def clean(self, text):
        return text.replace(self.NEW_PARAGRAPH_TOKEN, '').strip()

    def predict(self, texts, target_words=None, n=1, external_memory=None, 
        include_punctuation=False):
        # Prepare inputs
        input_idxes = [self.tokenizer.encode(text) for text in texts]
        if target_words is None:
            target_words = [None] * len(input_idxes)
            # target_words = [t.replace(self.SUBWORD_PREFIX, '') for t in target_words if t]

        # Pad token
        max_token_size = max([len(t) for t in input_idxes])
        for i, token_input in enumerate(input_idxes):
            for _ in range(max_token_size - len(token_input)):
                input_idxes[i].append(self.pad_id)

        target_poses = []
        if external_memory is None:  # First step or does not enable optimization
            for i, tokens in enumerate(input_idxes):
                target_poses.append(len(self.padding_text_idxes) + tokens.index(self.mask_id))
                input_idxes[i] = self.padding_text_idxes + tokens
        else:
            for i, tokens in enumerate(input_idxes):
                target_poses.append(tokens.index(self.mask_id))

        perm_masks = torch.zeros((len(input_idxes), len(input_idxes[0]), len(input_idxes[0])), dtype=torch.float)
        target_mappings = torch.zeros((len(input_idxes), 1, len(input_idxes[0])), dtype=torch.float)
        for i, target_pos in enumerate(target_poses):
            perm_masks[i][:, target_pos] = 1.0  # Mask the target word
            target_mappings[i, 0, target_pos] = 1.0

        # Convert to feature
        input_idxes = torch.tensor(input_idxes).to(self.device)
        perm_masks = perm_masks.to(self.device)
        target_mappings = target_mappings.to(self.device)

        # Prediction
        results = []
        with torch.no_grad():
            outputs = self.model(input_ids=input_idxes, perm_mask=perm_masks, target_mapping=target_mappings,
                mems=external_memory)

        # Selection
        for output, target_token in zip(outputs[0], target_words):
            target_token_logits = output[0]

            seed = {'temperature': self.temperature, 'top_k': self.top_k, 'top_p': self.top_p}
            target_token_logits = self.control_randomness(target_token_logits, seed)
            target_token_logits, target_token_idxes = self.filtering(target_token_logits, seed)
            if len(target_token_idxes) != 0:
                new_tokens = self.pick(target_token_logits, target_token_idxes, target_word=target_token, 
                    n=10, include_punctuation=include_punctuation)
                results.append([t[0] for t in new_tokens])
            else:
                results.append([''])

        return results
