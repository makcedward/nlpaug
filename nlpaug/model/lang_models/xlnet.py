import torch
from pytorch_transformers import XLNetTokenizer, XLNetLMHeadModel

from nlpaug.model.lang_models import LanguageModels


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

    def __init__(self, model_path='xlnet-base-cased', device='cuda'):
        super().__init__()
        self.model_path = model_path
        self.device = device

        self.tokenizer = XLNetTokenizer.from_pretrained(model_path)
        self.model = XLNetLMHeadModel.from_pretrained(model_path)

        self.padding_text_idxes = self.tokenizer.encode(self.PADDING_TEXT)

        self.model.to(device)
        self.model.eval()

    def predict(self, text, target_word=None, top_n=5):
        results = []

        # Convert to idxes
        input_idxes = self.tokenizer.encode(text)
        concatenated_idxes = self.padding_text_idxes + input_idxes
        target_pos = len(self.padding_text_idxes) + input_idxes.index(self.MASK_TOKEN_ID)

        # Convert feature
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

        # Get top_n candidate words
        logits, idxes = torch.topk(outputs[0][0][0], k=top_n + 20)
        for idx, logit in zip(idxes, logits):
            candidate_word = self.tokenizer.decode(idx.item())

            if target_word is not None and candidate_word.lower() == target_word.lower():
                continue

            candidate_value = logit.item()

            results.append(candidate_word)
            if len(results) >= top_n:
                break

        return results
