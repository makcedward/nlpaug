from nlpaug.augmenter.spectrogram import SpectrogramAugmenter
from nlpaug.util import Action
from nlpaug.model import TimeMasking


class TimeMaskingAug(SpectrogramAugmenter):
    def __init__(self, mask_factor, name='TimeMasking_Aug'):
        super(TimeMaskingAug, self).__init__(
            action=Action.SUBSTITUTE, name=name, aug_p=1, aug_min=0.3)

        self.model = self.get_model(mask_factor)

    def substitute(self, mel_spectrogram):
        return self.model.mask(mel_spectrogram)

    def get_model(self, mask_factor):
        return TimeMasking(mask_factor)
