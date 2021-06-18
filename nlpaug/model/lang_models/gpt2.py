import logging

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    # No installation required if not using this function
    pass

from nlpaug.model.lang_models import LanguageModels


class Gpt2(LanguageModels):
    # https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

    UNKNOWN_TOKEN = '<|endoftext|>'
    SUBWORD_PREFIX = 'Ġ'

    def __init__(self, model_path='gpt2', temperature=1.0, top_k=None, top_p=None, device=None, optimize=None, silence=True):
        super().__init__(device, temperature=temperature, top_k=top_k, top_p=top_p, optimize=optimize, silence=True)
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ModuleNotFoundError:
            raise ModuleNotFoundError('Missed transformers library. Install transfomers by `pip install transformers`')
            
        self.model_path = model_path

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.pad_id = 1 # No padding for GPT2, https://github.com/huggingface/transformers/issues/2630
        if silence:
            # Transformers thrown an warning regrading to weight initialization. It is expected
            orig_log_level = logging.getLogger('transformers.' + 'modeling_utils').getEffectiveLevel()
            logging.getLogger('transformers.' + 'modeling_utils').setLevel(logging.ERROR)
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            logging.getLogger('transformers.' + 'modeling_utils').setLevel(orig_log_level)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path)

        self.model.to(self.device)
        self.model.eval()

    def get_device(self):
        return str(self.model.device)

    def get_subword_prefix(self):
        return self.SUBWORD_PREFIX
        
    def id2token(self, _id):
        return self.tokenizer.decode(_id, clean_up_tokenization_spaces=True).strip()

    def predict(self, texts, target_words=None, n=1, external_memory=None, 
        include_punctuation=False):
        # Prepare inputs
        input_idxes = [self.tokenizer.encode(text) for text in texts]
        if target_words is None:
            target_words = [None] * len(input_idxes)
        mask_inputs = []

        # Pad token
        max_token_size = max([len(t) for t in input_idxes])
        for i, token_input in enumerate(input_idxes):
            mask_input = [1] * len(input_idxes[0])  # 1: are not masked, 0: masked token (for padding)

            for _ in range(max_token_size - len(token_input)):
                input_idxes[i].append(self.pad_id)
                mask_input.append(0)

            mask_inputs.append(mask_input)

        # Convert to feature
        input_idxes = torch.tensor(input_idxes).to(self.device)
        mask_inputs = torch.tensor(mask_inputs).to(self.device)

        # Prediction
        results = []
        with torch.no_grad():
            outputs = self.model(input_ids=input_idxes, attention_mask=mask_inputs, past_key_values=external_memory)

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
