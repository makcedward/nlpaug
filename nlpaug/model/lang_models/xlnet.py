# Source: https://arxiv.org/abs/1906.08237

try:
    import torch
    from transformers import XLNetTokenizer, XLNetLMHeadModel
except ImportError:
    # No installation required if not using this function
    pass

from nlpaug.model.lang_models import LanguageModels
from nlpaug.util.selection.filtering import *


class XlNet(LanguageModels):
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
                 device=None):
        super().__init__(device, temperature=temperature, top_k=top_k, top_p=top_p)
        self.model_path = model_path

        self.tokenizer = XLNetTokenizer.from_pretrained(model_path)
        self.model = XLNetLMHeadModel.from_pretrained(model_path)

        self.padding_text_idxes = self.tokenizer.encode(padding_text or self.PADDING_TEXT)

        self.model.to(self.device)
        self.model.eval()

    def id2token(self, _id):
        return self.tokenizer.decode(_id, clean_up_tokenization_spaces=True).strip()

    def clean(self, text):
        return text.replace(self.NEW_PARAGRAPH_TOKEN, '').strip()

    def predict(self, text, target_word=None, n=1):
        # Convert feature
        input_idxes = self.tokenizer.encode(text)
        concatenated_idxes = self.padding_text_idxes + input_idxes
        target_pos = len(self.padding_text_idxes) + input_idxes.index(self.MASK_TOKEN_ID)

        input_idxes = torch.tensor(concatenated_idxes).unsqueeze(0)
        perm_masks = torch.zeros((1, input_idxes.shape[1], input_idxes.shape[1]), dtype=torch.float)
        perm_masks[:, :, target_pos] = 1.0  # Mask the target word
        target_mappings = torch.zeros((1, 1, input_idxes.shape[1]), dtype=torch.float)
        target_mappings[0, 0, target_pos] = 1.0

        input_idxes = input_idxes.to(self.device)
        perm_masks = perm_masks.to(self.device)
        target_mappings = target_mappings.to(self.device)

        # Prediction
        with torch.no_grad():
            outputs = self.model(input_idxes, perm_mask=perm_masks, target_mapping=target_mappings)
        target_token_logits = outputs[0][0][0]  # XLNet return masked token only

        # Selection
        seed = {'temperature': self.temperature, 'top_k': self.top_k, 'top_p': self.top_p}
        target_token_logits = self.control_randomness(target_token_logits, seed)
        target_token_logits, target_token_idxes = self.filtering(target_token_logits, seed)

        results = self.pick(target_token_logits, target_word=target_word, n=n)
        return results