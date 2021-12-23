import logging

try:
    import torch
    from transformers import pipeline
except ImportError:
    # No installation required if not using this function
    pass

from nlpaug.model.lang_models import LanguageModels


class XSumTransformers(LanguageModels):
    def __init__(self, model_name="t5-base", tokenizer_name=None, min_length=10, max_length=20, 
        temperature=1.0, top_k=50, top_p=0.9, batch_size=32, device='cuda', silence=True):
        super().__init__(device, model_type=None, silence=silence)
        try:
            from transformers import pipeline
        except ModuleNotFoundError:
            raise ModuleNotFoundError('Missed transformers library. Install transfomers by `pip install transformers`')

        self.model_name = model_name
        self.tokenizer_name = model_name if tokenizer_name is None else tokenizer_name
        self.min_length = min_length
        self.max_length = max_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.batch_size = batch_size

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
        results = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                predict_result = self.model(texts[i:i+self.batch_size], 
                    min_length=self.min_length, 
                    max_length=self.max_length,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    num_workers=1)
                if isinstance(predict_result, list):
                    results.extend(predict_result)
                else:
                    results.append(predict_result)
        results = [r['summary_text'] for r in results]
        
        return results
