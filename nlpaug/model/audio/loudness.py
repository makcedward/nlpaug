import numpy as np

from nlpaug.model.audio import Audio


class Loudness(Audio):
    def manipulate(self, data, start_pos, end_pos, loudness_level):
        aug_data = data.copy()
        aug_data[start_pos:end_pos] = aug_data[start_pos:end_pos] * loudness_level

        return aug_data
