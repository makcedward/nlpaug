import logging

try:
    import torch
    from transformers import pipeline
except ImportError:
    # No installation required if not using this function
    pass

from nlpaug.model.lang_models import LanguageModels


class XSumTransformers(LanguageModels):
    def __init__(self, model_name="t5-base", tokenizer_name=None, min_length=10, max_length=20, device='cuda', silence=True):
        super().__init__(device, model_type=None, silence=silence)
        try:
            from transformers import pipeline
        except ModuleNotFoundError:
            raise ModuleNotFoundError('Missed transformers library. Install transfomers by `pip install transformers`')

        self.model_name = model_name
        self.tokenizer_name = model_name if tokenizer_name is None else tokenizer_name
        self.min_length = min_length
        self.max_length = max_length

        if silence:
            # Transformers thrown an warning regrading to weight initialization. It is expected
            orig_log_level = logging.getLogger('transformers.' + 'modeling_utils').getEffectiveLevel()
            logging.getLogger('transformers.' + 'modeling_utils').setLevel(logging.ERROR)

            device = self.convert_device(device)

            self.model = pipeline("summarization", model=self.model_name, tokenizer=self.tokenizer_name, 
                device=device, framework="pt")
            logging.getLogger('transformers.' + 'modeling_utils').setLevel(orig_log_level)

    def get_device(self):
        return str(self.model.device)

    def predict(self, texts, target_words=None, n=1):
        results = self.model(texts, min_length=self.min_length, max_length=self.max_length)
        results = [r['summary_text'] for r in results]
        
        return results
