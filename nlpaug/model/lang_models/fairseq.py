try:
    import torch
except ImportError:
    # No installation required if not using this function
    pass

from nlpaug.model.lang_models import LanguageModels
from nlpaug.util.selection.filtering import *


class Fairseq(LanguageModels):
    def __init__(self, from_model_name, from_model_checkpt, to_model_name, to_model_checkpt, tokenzier_name='moses', bpe_name='fastbpe', device='cuda'):
        super().__init__(device, temperature=None, top_k=None, top_p=None)

        try:
            import torch
            import fairseq
            self.device = 'cuda' if device is None and torch.cuda.is_available() else device
        except ImportError:
            raise ImportError('Missed torch, fairseq libraries. Install torch by following https://pytorch.org/get-started/locally/ and fairseq by '
                              'https://github.com/pytorch/fairseq')
        
        self.from_model_name = from_model_name
        self.from_model_checkpt = from_model_checkpt
        self.to_model_name = to_model_name
        self.to_model_checkpt = to_model_checkpt
        self.tokenzier_name = tokenzier_name
        self.bpe_name = bpe_name
        
        # TODO: enahnce to support custom model. https://github.com/pytorch/fairseq/tree/master/examples/translation
        self.from_model = torch.hub.load(
            github='pytorch/fairseq', model=from_model_name,
            checkpoint_file=from_model_checkpt,
            tokenizer=tokenzier_name, bpe=bpe_name)
        self.to_model = torch.hub.load(
            github='pytorch/fairseq', model=to_model_name,
            checkpoint_file=to_model_checkpt,
            tokenizer=tokenzier_name, bpe=bpe_name)

        self.from_model.eval()
        self.to_model.eval()
        if self.device == 'cuda':
            self.from_model.cuda()
            self.to_model.cuda()

    def predict(self, text):
        translated_text = self.from_model.translate(text)
        back_translated_text = self.to_model.translate(translated_text)
        
        return back_translated_text
