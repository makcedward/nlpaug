import numpy as np

from nlpaug.model.spectrogram import Spectrogram


class TimeMasking(Spectrogram):
    def __init__(self):
        super().__init__()

    def manipulate(self, data, t, t0):
        """
            From: https://arxiv.org/pdf/1904.08779.pdf,
            Time masking is applied so that t consecutive time steps
            [t0, t0 + t) are masked, where t is first chosen from a
            uniform distribution from 0 to the time mask parameter
            T, and t0 is chosen from [0, tau - t).
        """

        aug_data = data.copy()
        aug_data[:, t0:t0+t] = 0
        return aug_data
