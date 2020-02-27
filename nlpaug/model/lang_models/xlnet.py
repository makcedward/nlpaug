try:
    import torch
    from transformers import XLNetTokenizer, XLNetLMHeadModel
    # from transformers import AutoModel, AutoTokenizer
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
    MASK_TOKEN_ID = 6
    SUBWORD_PREFIX = '▁'
    NEW_PARAGRAPH_TOKEN = '<eop>'

    def __init__(self, model_path='xlnet-base-cased', temperature=1.0, top_k=None, top_p=None, padding_text=None,
                 optimize=None, device=None):
        super().__init__(device, temperature=temperature, top_k=top_k, top_p=top_p, optimize=optimize)
        self.model_path = model_path

        # self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # self.model = AutoModel.from_pretrained(model_path)
        # TODO: Evaluted to use mems in XLNet but the result is quite weird.
        self.optimize['external_memory'] = 0
        self.tokenizer = XLNetTokenizer.from_pretrained(model_path)
        self.model = XLNetLMHeadModel.from_pretrained(model_path, mem_len=self.optimize['external_memory'])

        self.padding_text_idxes = self.tokenizer.encode(padding_text or self.PADDING_TEXT)

        self.model.to(self.device)
        self.model.eval()

    def id2token(self, _id):
        return self.tokenizer.decode(_id, clean_up_tokenization_spaces=True).strip()

    def clean(self, text):
        return text.replace(self.NEW_PARAGRAPH_TOKEN, '').strip()

    def predict(self, text, target_word=None, n=1, external_memory=None):
        # Convert feature
        input_idxes = self.tokenizer.encode(text)

        if target_word is not None:
            target_word = target_word.replace(self.SUBWORD_PREFIX, '')

        if external_memory is None:  # First step or does not enable optimization
            target_pos = len(self.padding_text_idxes) + input_idxes.index(self.MASK_TOKEN_ID)
            input_idxes = torch.tensor(self.padding_text_idxes + input_idxes).unsqueeze(0)
        else:
            target_pos = input_idxes.index(self.MASK_TOKEN_ID)
            input_idxes = torch.tensor(input_idxes).unsqueeze(0)

        perm_masks = torch.zeros((1, input_idxes.shape[1], input_idxes.shape[1]), dtype=torch.float)
        perm_masks[:, :, target_pos] = 1.0  # Mask the target word
        target_mappings = torch.zeros((1, 1, input_idxes.shape[1]), dtype=torch.float)
        target_mappings[0, 0, target_pos] = 1.0

        input_idxes = input_idxes.to(self.device)
        perm_masks = perm_masks.to(self.device)
        target_mappings = target_mappings.to(self.device)

        # Prediction
        with torch.no_grad():
            outputs = self.model(input_ids=input_idxes, perm_mask=perm_masks, target_mapping=target_mappings,
                                 mems=external_memory)
        target_token_logits = outputs[0][0][0]  # XLNet return masked token only

        # Selection
        seed = {'temperature': self.temperature, 'top_k': self.top_k, 'top_p': self.top_p}
        target_token_logits = self.control_randomness(target_token_logits, seed)
        target_token_logits, target_token_idxes = self.filtering(target_token_logits, seed)
        if len(target_token_idxes) != 0:
            results = self.pick(target_token_logits, target_token_idxes, target_word=target_word, n=n)
        else:
            results = None

        results = (results,)
        if self.optimize['external_memory']:
            external_memory = outputs[1]
            results += (external_memory,)

        return results
