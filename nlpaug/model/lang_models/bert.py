import torch
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM

from nlpaug.model.lang_models import LanguageModels


class Bert(LanguageModels):
    START = '[CLS]'
    SEPARATOR = '[SEP]'
    MASK = '[MASK]'
    SUBWORD_PREFIX = '#'

    def __init__(self, model_path, tokenizer_path):
        super(Bert, self).__init__()
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path

        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.model = BertForMaskedLM.from_pretrained(model_path)

    def predict(self, input_tokens, target_word, top_n):
        results = []

        tokens = input_tokens.copy()
        tokens.insert(0, Bert.START)
        tokens.append(Bert.SEPARATOR)

        # Mask target word
        target_pos = tokens.index(target_word)
        target_idx = self.tokenizer.convert_tokens_to_ids([target_word])[0]
        tokens[target_pos] = Bert.MASK

        # Convert feature
        token_idxes = self.tokenizer.convert_tokens_to_ids(tokens)
        segments_idxes = [0] * len(tokens)
        tokens_features = torch.tensor([token_idxes])
        segments_features = torch.tensor([segments_idxes])

        # Predict target word
        predictions = self.model(tokens_features, segments_features)

        top_score_idx = target_idx
        for i in range(100):
            predictions[0, target_pos, top_score_idx] = -9999
            top_score_idx = torch.argmax(predictions[0, target_pos]).item()
            top_score_token = self.tokenizer.convert_ids_to_tokens([top_score_idx])[0]
            if top_score_token[0] != Bert.SUBWORD_PREFIX:
                results.append(top_score_token)
                if len(results) >= top_n:
                    break

        return results
