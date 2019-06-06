from nlpaug.augmenter.spectrogram import SpectrogramAugmenter
from nlpaug.util import Action
import nlpaug.model.spectrogram as nms


class FrequencyMaskingAug(SpectrogramAugmenter):
    def __init__(self, mask_factor, name='FrequencyMasking_Aug', verbose=0):
        super(FrequencyMaskingAug, self).__init__(
            action=Action.SUBSTITUTE, name=name, aug_p=1, aug_min=0.3, verbose=verbose)

        self.model = self.get_model(mask_factor)

    def substitute(self, mel_spectrogram):
        return self.model.mask(mel_spectrogram)

    def get_model(self, mask_factor):
        return nms.FrequencyMasking(mask_factor)
