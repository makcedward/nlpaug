import numpy as np

from nlpaug.model.audio import Audio


class Crop(Audio):
    def manipulate(self, data, start_pos, end_pos):
        aug_data = data.copy()
        aug_data = np.delete(aug_data, np.s_[start_pos:end_pos])
        return aug_data
