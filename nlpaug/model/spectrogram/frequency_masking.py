import numpy as np

from nlpaug.model.spectrogram import Spectrogram


class FrequencyMasking(Spectrogram):
    def __init__(self):
        super().__init__()

    def manipulate(self, data, f, f0, time_start, time_end):
        """
            https://arxiv.org/pdf/1904.08779.pdf, https://arxiv.org/pdf/2001.01401.pdf
            Frequency masking is applied so that f consecutive mel
            frequency channels [f0, f0 + f) are masked, where f is
            first chosen from a uniform distribution from 0 to the
            frequency mask parameter F, and f0 is chosen from
            [0, v - f). v is the number of mel frequency channels.
        """

        aug_data = data.copy()
        aug_data[f0:f0+f, time_start:time_end] = 0
        return aug_data
