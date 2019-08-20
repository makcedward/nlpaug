import numpy as np

from nlpaug.model.spectrogram import Spectrogram


class TimeMasking(Spectrogram):
    def __init__(self, mask_factor):
        super(TimeMasking, self).__init__()

        self.mask_factor = mask_factor

    def mask(self, data):
        """
            From: https://arxiv.org/pdf/1904.08779.pdf,
            Time masking is applied so that t consecutive time steps
            [t0, t0 + t) are masked, where t is first chosen from a
            uniform distribution from 0 to the time mask parameter
            T, and t0 is chosen from [0, tau - t).
        :return:
        """

        time_range = data.shape[1]
        self.t = np.random.randint(self.mask_factor)
        self.t0 = np.random.randint(time_range - self.t)

        augmented_mel_spectrogram = data.copy()
        augmented_mel_spectrogram[:, self.t0:self.t0+self.t] = 0
        return augmented_mel_spectrogram
