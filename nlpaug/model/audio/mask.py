import numpy as np

from nlpaug.model.audio import Audio


class Mask(Audio):
    def manipulate(self, data, start_pos, end_pos, mask_with_noise):
        if mask_with_noise:
            noise_data = np.random.randn(end_pos - start_pos)
        else:
            noise_data = np.zeros(end_pos - start_pos)

        aug_data = data.copy()
        aug_data[start_pos:end_pos] = noise_data

        return aug_data
