import logging

try:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except ImportError:
    # No installation required if not using this function
    pass

from nlpaug.model.lang_models import LanguageModels

import nlpaug.util.text.tokenizer as text_tokenizer


class T5(LanguageModels):
    # https://arxiv.org/pdf/1910.10683.pdf

    def __init__(self, model_path='t5-base', min_length=10, max_length=20, num_beam=3, no_repeat_ngram_size=3, 
        device='cuda', silence=True):
        super().__init__(device, temperature=None, top_k=None, top_p=None, silence=True)
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ModuleNotFoundError:
            raise ModuleNotFoundError('Missed transformers library. Install transfomers by `pip install transformers`')

        self.model_path = model_path
        self.min_length = min_length
        self.max_length = max_length
        self.num_beam = num_beam
        self.no_repeat_ngram_size = no_repeat_ngram_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if silence:
            # Transformers thrown an warning regrading to weight initialization. It is expected
            orig_log_level = logging.getLogger('transformers.' + 'modeling_utils').getEffectiveLevel()
            logging.getLogger('transformers.' + 'modeling_utils').setLevel(logging.ERROR)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            logging.getLogger('transformers.' + 'modeling_utils').setLevel(orig_log_level)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

        self.model.to(self.device)
        self.model.eval()

        self.text_prefix = 'summarize: '
        self.return_tensor = 'pt' # PyTorch
        self.early_stopping = True
        self.skip_special_token = True
        self.default_max_length_ratio = 0.5

    def predict(self, texts, n=1):
        # Convert to feature
        inputs = self.tokenizer([self.text_prefix + t for t in texts], padding='longest', 
            return_tensors=self.return_tensor)
        token_inputs = inputs['input_ids'].to(self.device)
        mask_inputs = inputs['attention_mask'].to(self.device)

        # Prediction
        min_length = min([len(text) for text in texts]) + len(self.text_prefix)
        min_length = self.get_min_length(min_length)

        max_length = max([len(text) for text in texts]) + len(self.text_prefix)
        max_length = self.get_max_length(max_length)

        results = []
        with torch.no_grad():
            outputs = self.model.generate(input_ids=token_inputs, attention_mask=mask_inputs,
                min_length=min_length, max_length=max_length, num_beams=self.num_beam,
                no_repeat_ngram_size=self.no_repeat_ngram_size)

        for target_token_ids in outputs:
            tokens = self.tokenizer.decode(target_token_ids, skip_special_tokens=self.skip_special_token)
            # Return full sentence only.
            for i in range(len(tokens)-1, -1, -1):
                if tokens[i] in text_tokenizer.SENTENCE_SEPARATOR:
                    results.append(tokens[:i+1])
                    break

        return results

    def get_min_length(self, min_length):
        return int(min_length * self.min_length) if self.min_length < 1 else self.min_length

    def get_max_length(self, max_length):
        if self.max_length < 1:
            return int(max_length * self.max_length)
        else:
            if max_length >= self.max_length:
                return int(max_length * self.default_max_length_ratio)
            else:
                return self.max_length
