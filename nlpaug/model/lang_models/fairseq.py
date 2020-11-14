import os

try:
    import torch
except ImportError:
    # No installation required if not using this function
    pass

from nlpaug.model.lang_models import LanguageModels
from nlpaug.util.selection.filtering import *


class Fairseq(LanguageModels):
    def __init__(self, from_model_name, from_model_checkpt, to_model_name, to_model_checkpt, 
        is_load_from_github=True, tokenzier_name='moses', bpe_name='fastbpe', 
        device='cuda'):
        super().__init__(device, temperature=None, top_k=None, top_p=None)

        try:
            import fairseq
            from fairseq.models.transformer import TransformerModel
        except ModuleNotFoundError:
            raise ModuleNotFoundError('Missed fairseq library. Install fairseq by https://github.com/pytorch/fairseq')
        
        self.from_model_name = from_model_name
        self.from_model_checkpt = from_model_checkpt
        self.to_model_name = to_model_name
        self.to_model_checkpt = to_model_checkpt
        self.is_load_from_github = is_load_from_github
        self.tokenzier_name = tokenzier_name
        self.bpe_name = bpe_name
        
        if is_load_from_github:
            self.from_model = torch.hub.load(
                'pytorch/fairseq', model=from_model_name,
                checkpoint_file=from_model_checkpt,
                tokenizer=tokenzier_name, bpe=bpe_name)
            self.to_model = torch.hub.load(
                'pytorch/fairseq', model=to_model_name,
                checkpoint_file=to_model_checkpt,
                tokenizer=tokenzier_name, bpe=bpe_name)
        else:
            try:
                self.from_model = TransformerModel.from_pretrained(
                    model_name_or_path=os.path.join(from_model_name, ''),
                    checkpoint_file=from_model_checkpt,
                    tokenizer=tokenzier_name, bpe=bpe_name)
            except TypeError:
                err_msg = 'Cannot load model from local path. You may check the following parameters are correct or not.'
                err_msg += ' Model Directory: ' + from_model_name
                err_msg += ', Checkpoint File Name: ' + from_model_checkpt
                err_msg += ', Tokenizer Name: ' + tokenzier_name
                err_msg += ', BPE Name: ' + bpe_name
                raise ValueError(err_msg)

            try:
                self.to_model = TransformerModel.from_pretrained(
                    model_name_or_path=os.path.join(to_model_name, ''),
                    checkpoint_file=to_model_checkpt,
                    tokenizer=tokenzier_name, bpe=bpe_name)
            except TypeError:
                err_msg = 'Cannot load model from local path. You may check the following parameters are correct or not.'
                err_msg += ' Model Directory: ' + to_model_name
                err_msg += ', Checkpoint File Name: ' + to_model_checkpt
                err_msg += ', Tokenizer Name: ' + tokenzier_name
                err_msg += ', BPE Name: ' + bpe_name
                raise ValueError(err_msg)

        self.from_model.eval()
        self.to_model.eval()
        if self.device == 'cuda':
            self.from_model.cuda()
            self.to_model.cuda()

    def predict(self, text):
        translated_text = self.from_model.translate(text, verbose=False)
        back_translated_text = self.to_model.translate(translated_text, verbose=False)
        
        return back_translated_text
