import numpy as np

from nlpaug.model.spectrogram import Spectrogram


class Loudness(Spectrogram):
    def __init__(self):
        super().__init__()

    def manipulate(self, data, loudness_level, time_start, time_end):
        # https://arxiv.org/pdf/2001.01401.pdf
        aug_data = data.copy()
        aug_data[:, time_start:time_end] = aug_data[:, time_start:time_end] * loudness_level * 1000
        return aug_data
