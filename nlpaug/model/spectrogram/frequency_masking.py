import numpy as np

from nlpaug.model.spectrogram import Spectrogram


class FrequencyMasking(Spectrogram):
    def __init__(self, mask_factor):
        super(FrequencyMasking, self).__init__()

        self.mask_factor = mask_factor

    def mask(self, data):
        """
            From: https://arxiv.org/pdf/1904.08779.pdf,
            Frequency masking is applied so that f consecutive mel
            frequency channels [f0, f0 + f) are masked, where f is
            first chosen from a uniform distribution from 0 to the
            frequency mask parameter F, and f0 is chosen from
            [0, v - f). v is the number of mel frequency channels.
        :return:
        """
        v  = data.shape[0]
        self.f = np.random.randint(self.mask_factor)
        self.f0 = np.random.randint(v - self.f)

        augmented_mel_spectrogram = data.copy()
        augmented_mel_spectrogram[self.f0:self.f0+self.f, :] = 0
        return augmented_mel_spectrogram
